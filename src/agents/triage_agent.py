"""分诊 Agent：确定性红旗优先，其次才使用结构化大模型判断。

LangGraph 调用 :func:`triage_node` 时，本模块先执行本地安全规则。命中红旗后直接
返回紧急结果，不调用模型；未命中时可按配置调用模型，并在模型关闭、调用失败或
输出不合法时给出保守的规则降级结果。节点不连接任何数据库。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT
from src.config.settings import settings
from src.graph.state import MedicalState
from src.models.schemas import MedicalError, PatientProfile, TriageResult
from src.services.llm_factory import get_chat_model
from src.tools.safety import RedFlagMatch, assess_red_flag_details


_GENERAL_DEPARTMENT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("孕", "阴道", "月经", "下腹"), "妇产科"),
    (("儿童", "小儿", "宝宝", "婴儿"), "儿科"),
    (("咳嗽", "咳痰", "发热", "发烧", "气喘"), "呼吸内科"),
    (("腹痛", "腹泻", "呕吐", "胃痛", "便秘"), "消化内科"),
    (("头痛", "头晕", "肢体麻木", "手抖"), "神经内科"),
    (("皮疹", "瘙痒", "红疹", "脱皮"), "皮肤科"),
    (("关节", "骨痛", "扭伤", "腰痛"), "骨外科"),
    (("眼", "视力", "眼睛"), "眼科"),
    (("耳", "鼻", "咽", "喉"), "耳鼻喉科"),
)

# 调用结果会被自动缓存下来。下次用相同参数再调用时，不会真的执行函数体，而是直接把上次记下的结果返回
@lru_cache(maxsize=4)
def load_standard_departments(graph_nodes_path: str) -> frozenset[str]:
    """从 离线医学数据清洗、实体对齐与索引构建 图节点产物中提取标准科室词表。

    文件缺失、坏行或不可读时返回空集合，调用方随后使用通用科室建议。结果会被
    缓存，避免每次患者请求都扫描较大的 JSONL 文件。本函数只读本地产物，不访问
    MySQL、Neo4j 或 Chroma。
    """
    departments: set[str] = set()
    path = Path(graph_nodes_path)
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if item.get("entity_type") == "department" and item.get("name"):
                    departments.add(str(item["name"]).strip())
    except OSError:
        return frozenset()
    return frozenset(departments)


def _department_vocabulary() -> frozenset[str]:
    """取得当前 离线医学数据清洗、实体对齐与索引构建 产物中的标准科室名称；无产物时安全返回空集合。"""

    path = settings.data_dir / "processed" / "graph_nodes.jsonl"
    return load_standard_departments(str(path))


def _fallback_department(chief_complaint: str, profile: PatientProfile) -> str:
    """在不调用模型时按宽泛症状选择科室，不把匹配结果解释成诊断。"""

    vocabulary = _department_vocabulary()
    candidates: list[str] = []
    if profile.pregnancy:
        candidates.append("妇产科")
    if profile.age is not None and profile.age < 14:
        candidates.append("儿科")
    for keywords, department in _GENERAL_DEPARTMENT_RULES:
        if any(keyword in chief_complaint for keyword in keywords):
            candidates.append(department)
    # extend 是“拆开加”，append 是“整个塞”。
    candidates.extend(("普通内科", "全科医学科", "内科"))
    if vocabulary:
        return next((item for item in candidates if item in vocabulary), "待预问诊后确定")
    # 没词表时（vocabulary 为空）：没法过滤，就只能信规则排序，直接赌第一个 candidates[0]
    return candidates[0]


def _emergency_result(matches: list[RedFlagMatch]) -> TriageResult:
    """把确定性规则明细转换为强制终止普通预问诊的结果。"""

    # dict.fromkeys 得到一个按首次出现顺序排列、没有重复的 label 列表
    red_flags = list(dict.fromkeys(match.label for match in matches))
    return TriageResult(
        urgency="emergency",
        recommended_departments=["急诊科"],
        red_flags=red_flags,
        rationale="确定性安全规则检测到需立即线下医疗评估的风险信号。",
        should_continue_preconsult=False,
    )


def _rule_fallback_result(
    chief_complaint: str,
    profile: PatientProfile,
    *,
    model_failed: bool,
) -> TriageResult:
    """生成无模型时的合法分诊结果。

    模型只是按配置关闭时维持常规预问诊；模型原本启用却失败时采用更保守的
    ``urgent``，提示尽快由线下医务人员评估，但仍允许系统继续收集病史。
    """

    return TriageResult(
        urgency="urgent" if model_failed else "routine",
        recommended_departments=[_fallback_department(chief_complaint, profile)],
        red_flags=[],
        rationale=(
            "模型暂时不可用；未命中确定性红旗规则，建议尽快线下评估并继续采集病史。"
            if model_failed
            else "未命中确定性红旗规则，先按通用科室建议继续采集症状和病史。"
        ),
        should_continue_preconsult=True,
    )


def _validate_model_result(raw_result: Any, fallback_department: str) -> TriageResult:
    """模型可以靠语义判断紧急，但不能谎称"命中红旗规则"。"""

    result = (
        raw_result # 大模型直接输出的结果
        # 将模型输出转化为固定格式
        if isinstance(raw_result, TriageResult)
        else TriageResult.model_validate(raw_result)
    )
    departments = [item.strip() for item in result.recommended_departments if item.strip()]
    vocabulary = _department_vocabulary()
    if vocabulary:
        departments = [item for item in departments if item in vocabulary]
    if not departments:
        departments = [fallback_department]

    # model_dump()：把 Pydantic 模型对象 转成 Python 字典（输出/序列化）
    # model_validate()：把 原始数据（字典、JSON 或其他对象）转成 Pydantic 模型对象（输入/反序列化）

    # 标签；模型仍可根据整体语义判为 emergency，但不能冒充规则命中
    # 重新经过 Pydantic 校验，而不是直接 model_copy，确保调整后的结果仍满足
    # “紧急状态不得继续普通预问诊”等共享模型不变量。
    return TriageResult.model_validate(
        result.model_dump()
        | {
            "recommended_departments": list(dict.fromkeys(departments)),
            # red_flags 的唯一可信来源是前置规则。无规则命中时清空模型可能自行生成的
            "red_flags": [], # 强制清空红旗标签
            # 由紧急等级反推：emergency → False（不继续问诊）
            "should_continue_preconsult": result.urgency != "emergency",
        }
    )


def _invoke_model(chief_complaint: str, profile: PatientProfile) -> TriageResult:
    """调用支持 Pydantic 结构化输出的模型并验证结果。"""

    # 保证提示词每次一致——set/frozenset 无序
    vocabulary = sorted(_department_vocabulary())
    model = get_chat_model(temperature=0).with_structured_output(TriageResult)
    messages = [
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(
            content=TRIAGE_USER_PROMPT.format(
                chief_complaint=chief_complaint,
                patient_profile=json.dumps(profile.model_dump(), ensure_ascii=False),
                department_vocabulary="、".join(vocabulary) if vocabulary else "通用科室名称",
            )
        ),
    ]
    raw_result = model.invoke(messages)
    return _validate_model_result(
        raw_result,
        fallback_department=_fallback_department(chief_complaint, profile),
    )


def triage_node(state: MedicalState) -> dict[str, Any]:
    """根据共享状态返回分诊字段、审计日志及可选错误记录。

    输入必须包含 ``chief_complaint``，患者资料缺省时按未知处理。节点只返回发生
    变化的字段，不原地修改共享状态。工作流随后读取
    ``should_continue_preconsult`` 决定紧急结束或进入预问诊。
    """
    # 读取患者主诉并清除文本两端的空格
    chief_complaint = str(state.get("chief_complaint", "")).strip()
    # 校验患者资料
    profile = PatientProfile.model_validate(state.get("patient_profile", {}))
    # 主诉命中的可解释红旗规则
    matches = assess_red_flag_details(chief_complaint, profile)
    errors: list[dict[str, Any]] = []

    # matches 非空 → 说明主诉里命中了确定性红旗规则（大出血、意识障碍等)
    if matches:
        # 调 _emergency_result(matches) 生成紧急结果：
        result = _emergency_result(matches)
        decision_source = "deterministic_red_flag_rules"
    elif not settings.llm_enabled:
        # 模型被配置关闭,生成规则降级结果
        result = _rule_fallback_result(chief_complaint, profile, model_failed=False)
        decision_source = "rules_only_llm_disabled"
    else:
        try:
            result = _invoke_model(chief_complaint, profile)
            decision_source = "structured_llm"
        except Exception:
            # 不把 SDK 异常、请求内容或密钥写入状态；安全、错误处理与可观测性 后续可统一接入详细日志。
            result = _rule_fallback_result(chief_complaint, profile, model_failed=True)
            decision_source = "rule_fallback_llm_error"
            errors.append(
                MedicalError(
                    category="llm",
                    code="triage_model_unavailable",
                    message="分诊模型暂时不可用，已采用保守规则结果。",
                    retryable=True,
                    node="triage",
                ).model_dump()
            )

    result_data = result.model_dump()
    update: dict[str, Any] = {
        "triage_result": result_data,
        "conversation_status": "triaging",
        "audit_log": [
            {
                "node": "triage",
                "decision_source": decision_source,
                "urgency": result.urgency,
                "should_continue_preconsult": result.should_continue_preconsult,
                "rationale": result.rationale,
                "matched_rules": [
                    {"code": item.code, "label": item.label, "matched_term": item.matched_term}
                    for item in matches
                ],
                "recommended_departments": result.recommended_departments,
            }
        ],
    }
    if errors:
        update["errors"] = errors
    return update
