"""预问诊 Agent：逐轮提取患者信息、决定检索并生成一个追问。

LangGraph 每次进入 :func:`preconsult_node` 时，本模块只读取共享状态，不直接访问
数据库。模型可用时使用 ``PreconsultDecision`` 约束输出；模型关闭或调用失败时，
使用确定性中文规则完成同样的槽位合并和追问决策，保证问诊仍可继续。

患者明确说“没有、无、否认”时，否定事实会以“否认……”保存在对应列表槽位，
供 预问诊摘要与病历草稿 摘要 Agent 区分阴性发现和未知信息。图谱证据只用于优化问题措辞，绝不会
被写成患者已经确诊的疾病。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agents.prompts import PRECONSULT_SYSTEM_PROMPT, PRECONSULT_USER_PROMPT
from src.config.settings import settings
from src.graph.state import MedicalState
from src.models.schemas import (
    ConsultationSlots,
    MedicalError,
    PreconsultDecision,
    RetrievalIntent,
)
from src.services.llm_factory import get_chat_model


# 列表槽位采用“追加并去重”，不会因某轮模型漏提字段而清空已经确认的事实。
_LIST_FIELDS = {
    "symptom",
    "accompanying_symptoms",
    "medical_history",
    "medication_history",
    "allergy_history",
    "special_population",
}
# 先取得 ConsultationSlots 的所有字段，再减去列表字段，剩下的就是标量字段(新值覆盖旧值)
_SCALAR_FIELDS = set(ConsultationSlots.model_fields) - _LIST_FIELDS

# 持续时间表
_DURATION_PATTERN = re.compile(
    r"(?:大约|约)?\d+(?:\.\d+)?\s*(?:分钟|小时|天|周|星期|个月|月|年)(?:前|左右)?|"
    r"今天|昨天|前天|刚刚|刚才|近期|最近|从小|多年"
)
# 严重程度表达式
_SEVERITY_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*分|轻微|较轻|一般|中等|明显|严重|剧烈|难以忍受|"
    r"影响(?:睡眠|进食|工作|活动|日常生活)|不影响(?:睡眠|进食|工作|活动|日常生活)"
)
# 部位词表
_LOCATION_TERMS = (
    "头部", "额头", "太阳穴", "后脑", "眼眶", "胸口", "胸部", "上腹", "下腹",
    "腹部", "胃部", "腰部", "背部", "咽部", "喉咙", "左侧", "右侧", "双侧",
    "全身", "关节", "膝盖", "肩部", "颈部",
)
# 症状性质词表
_CHARACTERISTIC_TERMS = (
    "胀痛", "刺痛", "隐痛", "钝痛", "绞痛", "灼痛", "跳痛", "压迫感", "闷痛",
    "阵发", "持续", "间歇", "干咳", "有痰", "瘙痒", "麻木", "酸痛",
)
# 常见伴随症状表
_ACCOMPANYING_TERMS = (
    "发热", "发烧", "头晕", "恶心", "呕吐", "腹泻", "咳嗽", "咳痰", "鼻塞",
    "流涕", "乏力", "心悸", "胸闷", "呼吸困难", "皮疹", "出血", "畏光",
)
# 否定词
_NEGATION_PATTERN = re.compile(r"(?:没有|无|否认|未出现|不伴|并无|没|不是)")


# 标准问题表
_QUESTIONS: dict[str, str] = {
    "onset": "这次不适是突然出现的，还是逐渐出现的？",
    "duration": "这种不适从什么时候开始，持续多久了？",
    "severity": "目前症状有多严重，是否影响睡眠、进食或日常活动？",
    "location": "不适主要位于身体哪个部位？",
    "characteristics": "这种不适具体是什么感觉或性质？",
    "aggravating_or_relieving_factors": "什么情况会让症状加重或缓解？",
    "accompanying_symptoms": "除主诉外，是否还伴有其他不适？没有也请明确说明。",
    "medical_history": "您是否有相关既往疾病或手术史？没有也请明确说明。",
    "medication_history": "近期或长期正在使用哪些药物？没有也请明确说明。",
    "allergy_history": "您是否有药物或食物过敏？没有也请明确说明。",
}
# 每个条目只问一个核心主题。优先级会按主诉类型和分诊紧迫度动态调整。
_DEFAULT_PRIORITY = (
    "onset", "duration", "severity", "accompanying_symptoms", "location",
    "characteristics", "aggravating_or_relieving_factors", "medical_history",
    "medication_history", "allergy_history",
)
_PAIN_PRIORITY = (
    "location", "onset", "duration", "severity", "accompanying_symptoms",
    "characteristics", "aggravating_or_relieving_factors", "medical_history",
    "medication_history", "allergy_history",
)
_URGENT_PRIORITY = (
    "onset", "duration", "severity", "accompanying_symptoms", "location",
    "characteristics", "aggravating_or_relieving_factors", "medical_history",
    "medication_history", "allergy_history",
)


def _normalise_text(value: Any) -> str:
    """把消息内容转换为适合规则提取的纯文本。"""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and item.get("type") in {"text", "input_text"}:
                parts.append(str(item.get("text", "")))
        return " ".join(part.strip() for part in parts if part.strip())
    # 其他类型返回空字符串
    return ""


def latest_patient_text(state: Mapping[str, Any]) -> str:
    """从消息历史中读取最新患者文本；没有消息时退回首次主诉。"""

    # reversed(...)：从最后一条消息向前找，因为最新患者消息通常在末尾。
    for message in reversed(state.get("messages", []) or []):
        # 消息确实是 HumanMessage 对象 | 某些序列化恢复的消息有 type="human"。
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            # 提取文本
            text = _normalise_text(getattr(message, "content", ""))
            if text:
                return text
    # 没有消息历史，则退回首次主诉
    return str(state.get("chief_complaint", "")).strip()


def _question_field(question: str | None) -> str | None:
    """根据上一轮系统问题识别患者当前回答所对应的槽位。"""

    if not question:
        return None
    keyword_map = (
        ("onset", ("突然", "逐渐")),
        ("duration", ("什么时候", "持续多久", "多长时间")),
        ("severity", ("多严重", "严重程度", "影响睡眠", "日常活动")),
        ("location", ("哪个部位", "哪里", "位置")),
        ("characteristics", ("什么感觉", "性质")),
        ("aggravating_or_relieving_factors", ("加重", "缓解", "诱因")),
        ("accompanying_symptoms", ("伴有", "其他不适")),
        ("medical_history", ("既往", "疾病", "手术史")),
        ("medication_history", ("使用哪些药", "用药", "药物")),
        ("allergy_history", ("过敏",)),
    )
    return next((field for field, words in keyword_map if any(word in question for word in words)), None)


def _negative_value(field: str) -> str:
    """为宽泛的“没有”回答生成含义明确的阴性事实。"""

    return {
        "accompanying_symptoms": "否认其他伴随症状",
        "medical_history": "否认相关既往病史",
        "medication_history": "否认近期及长期用药",
        "allergy_history": "否认已知过敏史",
    }[field]


def _negative_list_values(field: str, text: str) -> list[str]:
    """把否定回答转换为适合摘要的阴性事实，同时尽量保留具体症状。"""

    if field == "accompanying_symptoms":
        specific = [f"否认{term}" for term in _ACCOMPANYING_TERMS if term in text]
        if specific:
            return list(dict.fromkeys(specific))
    return [_negative_value(field)]


def _term_is_negated(text: str, term: str) -> bool:
    """判断某个症状词所在的局部短句是否被否定。

    否定词只作用于最近一个逗号、分号或“但”之后的局部内容，避免把
    “没有发热，但有呕吐”错误理解成同时否认发热和呕吐。
    """

    # 同一症状被多次提及时采用最后一次陈述，例如“之前没有发热，后来发热了”
    # 应以患者更新后的阳性状态为准。
    position = text.rfind(term)
    # 词不存在
    if position < 0:
        return False
    clause_start = max(
        # 在 term 之前找“最近的一个分隔符”的位置
        (text.rfind(separator, 0, position) for separator in ("，", ",", "。", "；", ";", "但")),
        default=-1,
    )
    # 截取 term 之前、最近分隔符之后的那一小段局部内容。
    prefix = text[clause_start + 1:position]
    # 只检查前面最近八个字符是否包含否定词
    return bool(_NEGATION_PATTERN.search(prefix[-8:]))


def _extract_accompanying_facts(text: str) -> list[str]:
    """逐个提取伴随症状及其阴阳性，保留患者同句中的混合表达。"""

    facts: list[str] = []
    for term in _ACCOMPANYING_TERMS:
        if term not in text:
            continue
        facts.append(f"否认{term}" if _term_is_negated(text, term) else term)
    return list(dict.fromkeys(facts))


def extract_rule_slot_updates(
    text: str,
    *,
    current_question: str | None = None,
    chief_complaint: str = "",
) -> dict[str, Any]:
    """使用确定性中文规则从一轮患者文本中提取槽位更新。

    规则先理解“这段回答是在回答哪个问题”，再补充文本中明显出现的时长、程度、
    部位和伴随症状。只返回本轮发现的字段，之后再与旧槽位合并。
    """
    # 清洗空白和标点
    cleaned = re.sub(r"\s+", " ", text).strip(" ，,。；;！!")
    if not cleaned:
        return {}
    # updates 只保存本轮提取出的变化
    updates: dict[str, Any] = {}
    # 确定这次在回答什么内容
    answered_field = _question_field(current_question)
    if answered_field in _LIST_FIELDS:
        # 回答伴随症状
        if answered_field == "accompanying_symptoms":
            accompanying_facts = _extract_accompanying_facts(cleaned)
            if accompanying_facts:
                updates[answered_field] = accompanying_facts
            # 没有识别出具体症状，但存在否定词
            elif _NEGATION_PATTERN.search(cleaned):
                updates[answered_field] = _negative_list_values(answered_field, cleaned)
            else:
                updates[answered_field] = [cleaned]
        # 回答既往史、用药史或过敏史
        elif answered_field != "symptom" and _NEGATION_PATTERN.search(cleaned):
            updates[answered_field] = _negative_list_values(answered_field, cleaned)
        else:
            updates[answered_field] = [cleaned]
    # 包含持续时间 严重程度 位置 症状性质
    elif answered_field in _SCALAR_FIELDS: 
        updates[answered_field] = cleaned

    duration = _DURATION_PATTERN.search(cleaned)
    if duration and answered_field != "duration":
        # duration.group(0)：返回正则完整匹配到的字符串
        updates["duration"] = duration.group(0).replace(" ", "")
    if any(word in cleaned for word in ("突然", "骤然", "一下子")):
        updates["onset"] = "突然出现"
    elif any(word in cleaned for word in ("逐渐", "慢慢", "渐渐")):
        updates["onset"] = "逐渐出现"

    severity = _SEVERITY_PATTERN.search(cleaned)
    if severity and answered_field != "severity":
        updates["severity"] = severity.group(0).replace(" ", "")

    locations = [term for term in _LOCATION_TERMS if term in cleaned]
    if locations:
        updates["location"] = "、".join(dict.fromkeys(locations))

    characteristics = [term for term in _CHARACTERISTIC_TERMS if term in cleaned]
    if characteristics:
        updates["characteristics"] = "、".join(dict.fromkeys(characteristics))

    # 更新symptom"
    accompanying_facts = _extract_accompanying_facts(cleaned)
    # 回答伴随症状问题时同时保留阳性与阴性；首次主诉只把阳性词作为症状。
    if accompanying_facts:
        if answered_field == "accompanying_symptoms":
            updates["accompanying_symptoms"] = accompanying_facts
        # 首次咨询
        elif not current_question:
            positive_facts = [item for item in accompanying_facts if not item.startswith("否认")]
            if positive_facts:
                updates["symptom"] = positive_facts
    # 没有提取到明确症状词，就把完整主诉作为症状。
    if not current_question and chief_complaint.strip():
        # symptom 已经存在：不覆盖; 不存在：写入主诉。
        updates.setdefault("symptom", [chief_complaint.strip()])
    return updates


def _as_clean_list(value: Any) -> list[str]:
    """把模型或规则的列表值清洗为非空、保持顺序且不重复的字符串列表。"""

    items: Iterable[Any] = value if isinstance(value, (list, tuple, set)) else [value]
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def merge_consultation_slots(
    current: ConsultationSlots,
    updates: Mapping[str, Any],
) -> ConsultationSlots:
    """把本轮槽位更新安全地合并进旧槽位，并重新执行 Pydantic 校验。

    未知字段会被忽略，列表采用去重追加，空值不会擦除旧数据；标量只有在给出
    非空新值时才覆盖。这是“多轮不会清空旧信息”的核心保证。
    """
    # 把旧 Pydantic 模型复制成新字典
    merged = current.model_dump()
    for field, value in updates.items():
        if field in _LIST_FIELDS:
            additions = _as_clean_list(value)
            if additions:
                # *merged[field] 和 *additions 是列表展开后拼接去重
                merged[field] = list(dict.fromkeys([*merged[field], *additions]))
        elif field in _SCALAR_FIELDS and value is not None and str(value).strip():
            merged[field] = str(value).strip()
    return ConsultationSlots.model_validate(merged)


def _priority_for(state: Mapping[str, Any], slots: ConsultationSlots) -> tuple[str, ...]:
    """依据分诊等级和主诉类型拼接主诉和症状并返回询问优先级。"""

    triage = state.get("triage_result") or {}
    if triage.get("urgency") == "urgent":
        return _URGENT_PRIORITY
    complaint = " ".join([str(state.get("chief_complaint", "")), *slots.symptom])
    if any(word in complaint for word in ("痛", "疼", "不适", "麻木")):
        return _PAIN_PRIORITY
    return _DEFAULT_PRIORITY


def _slot_is_filled(slots: ConsultationSlots, field: str) -> bool:
    """统一判断列表和标量槽位是否已经得到患者的明确回答。"""

    return bool(getattr(slots, field))


def _evidence_symptom(state: Mapping[str, Any], slots: ConsultationSlots) -> str | None:
    """从图谱证据中挑一个尚未由患者确认的症状，仅用于定向追问。"""

    # “否认发热”也代表发热这个字段已经问清，不能被图谱证据诱导后重复询问。
    known = set(slots.symptom) | {
        item.removeprefix("否认") for item in slots.accompanying_symptoms
    }
    for item in state.get("retrieved_evidence", []) or []:
        candidates = (
            (item.get("source_entity"), item.get("source_entity_type")),
            (item.get("target_entity"), item.get("target_entity_type")),
        )
        for name, entity_type in candidates:
            if entity_type == "symptom" and name and name not in known:
                return str(name)
    return None


def next_missing_question(
    slots: ConsultationSlots,
    *,
    state: Mapping[str, Any] | None = None,
) -> str | None:
    """按当前风险和主诉寻找第一个缺失槽位，每轮只返回一个问题。"""

    active_state = state or {}
    for field in _priority_for(active_state, slots):
        if _slot_is_filled(slots, field):
            continue
        if field == "accompanying_symptoms":
            # 利用图谱证据生成更有针对性的问题
            evidence_symptom = _evidence_symptom(active_state, slots)
            if evidence_symptom:
                return (
                    f"为了补全症状信息，请确认是否伴有“{evidence_symptom}”？"
                    "这只是信息采集，不代表诊断。"
                )
        return _QUESTIONS[field]
    return None


def _information_is_sufficient(
    slots: ConsultationSlots,
    state: Mapping[str, Any],
) -> bool:
    """判断是否已经具备进入摘要所需的最小核心信息。

    所有主诉至少需要时间、严重程度和伴随症状；疼痛类主诉还需要部位，紧急度为
    urgent 时还要求起病方式。既往史、用药和过敏若患者主动提供仍会保留，但不应
    为逐项填满所有槽位而无限延长预问诊。
    """

    required = {"duration", "severity", "accompanying_symptoms"}
    complaint = " ".join([str(state.get("chief_complaint", "")), *slots.symptom])
    if any(word in complaint for word in ("痛", "疼", "不适", "麻木")):
        # 疼痛类提问额外要求
        required.add("location")
    if (state.get("triage_result") or {}).get("urgency") == "urgent":
        # urgent 分诊额外要求
        required.add("onset")
    return bool(slots.symptom) and all(_slot_is_filled(slots, field) for field in required)


def _select_question(
    deterministic_question: str | None,
    decision: PreconsultDecision | None,
    *,
    previous_question: str | None,
) -> str | None:
    """在安全边界内采用模型措辞，否则使用确定性问题。

    模型问题必须与规则识别出的同一个缺失槽位一致、只含一个问号且不能重复上一
    问。这样既让结构化模型参与追问，又不会让它跳过已经验证的槽位策略。
    """

    candidate = decision.next_question.strip() if decision and decision.next_question else ""
    if not deterministic_question or not candidate or candidate == previous_question:
        return deterministic_question
    if candidate.count("？") + candidate.count("?") > 1:
        return deterministic_question
    if _question_field(candidate) != _question_field(deterministic_question):
        return deterministic_question
    return candidate


def _retrieval_signature(intent: RetrievalIntent, entities: list[str]) -> str:
    """生成稳定检索签名，用于识别已经尝试过的相同请求。"""
    # 先去重、排序，再生成
    return f"{intent}|{'|'.join(sorted(set(entities)))}"


def _requested_signatures(state: Mapping[str, Any]) -> set[str]:
    """从追加型审计日志恢复历史请求"""

    return {
        str(item["retrieval_signature"])
        for item in state.get("audit_log", []) or []
        if item.get("node") == "preconsult" and item.get("retrieval_signature")
    }


def _previously_requested_entities(
    state: Mapping[str, Any],
    intent: RetrievalIntent,
) -> set[str]:
    """从稳定签名中恢复某类检索已覆盖的实体名称。"""

    prefix = f"{intent}|"
    entities: set[str] = set()
    for signature in _requested_signatures(state):
        if signature.startswith(prefix):
            entities.update(item for item in signature[len(prefix):].split("|") if item)
    return entities


def _model_retrieval_request(
    decision: PreconsultDecision | None,
    state: Mapping[str, Any],
) -> tuple[RetrievalIntent, list[str]] | None:
    """只接受有患者槽位或图谱证据支撑的模型检索实体。"""

    if not decision or not decision.need_graph_retrieval or not decision.retrieval_intent:
        return None
    slots = ConsultationSlots.model_validate(state.get("consultation_slots", {}))
    # 从已有证据建立“实体名称 → 实体类型”映射
    evidence_entity_types = {
        str(value): entity_type
        for item in state.get("retrieved_evidence", []) or []
        for value, entity_type in (
            (item.get("source_entity"), item.get("source_entity_type")),
            (item.get("target_entity"), item.get("target_entity_type")),
        )
        if value
    }
    # 只允许使用图谱证据中类型为 disease 的实体
    if decision.retrieval_intent == "disease_to_check":
        allowed = {
            entity for entity, entity_type in evidence_entity_types.items()
            if entity_type == "disease"
        }
    else:
        # symptom_to_department 和 symptom_to_diseas 意图的起点必须是患者已确认的症状。
        allowed = set(slots.symptom)
    entities = [item for item in _as_clean_list(decision.retrieval_entities) if item in allowed]
    return (decision.retrieval_intent, entities) if entities else None


def _choose_retrieval_request(
    state: Mapping[str, Any],
    slots: ConsultationSlots,
    decision: PreconsultDecision | None,
) -> tuple[RetrievalIntent, list[str]] | None:
    """产生结构化检索请求，且不重复历史上已经尝试过的相同请求。"""
    # 先尝试使用模型请求
    candidate = _model_retrieval_request(decision, state)
    if candidate is None and slots.symptom:
        # 找出历史已经查询的症状
        covered = _previously_requested_entities(state, "symptom_to_department")
        # 遍历证据的来源和目标实体，只把 entity_type == "symptom" 的名称加入 covered
        covered.update(
            str(value)
            for item in state.get("retrieved_evidence", []) or []
            for value, entity_type in (
                (item.get("source_entity"), item.get("source_entity_type")),
                (item.get("target_entity"), item.get("target_entity_type")),
            )
            if value and entity_type == "symptom"
        )
        # 只保留新症状
        uncovered = [entity for entity in slots.symptom if entity not in covered]
        if uncovered:
            candidate = ("symptom_to_department", uncovered)
    if candidate is None:
        return None
    intent, entities = candidate
    clean_entities = _as_clean_list(entities)
    signature = _retrieval_signature(intent, clean_entities)
    # 清洗实体并生成签名（实体为空或相同签名已经出现)
    if not clean_entities or signature in _requested_signatures(state):
        return None
    return intent, clean_entities


def _invoke_model(state: Mapping[str, Any], latest_text: str) -> PreconsultDecision:
    """调用大模型生成受 ``PreconsultDecision`` 约束的结构化结果。"""

    model = get_chat_model(temperature=0).with_structured_output(PreconsultDecision)
    messages: list[BaseMessage] = [
        SystemMessage(content=PRECONSULT_SYSTEM_PROMPT),
        HumanMessage(
            content=PRECONSULT_USER_PROMPT.format(
                chief_complaint=state.get("chief_complaint", ""),
                latest_patient_message=latest_text,
                current_question=state.get("current_question") or "首次信息采集",
                consultation_slots=json.dumps(state.get("consultation_slots", {}), ensure_ascii=False),
                triage_result=json.dumps(state.get("triage_result") or {}, ensure_ascii=False),
                retrieved_evidence=json.dumps(state.get("retrieved_evidence", []), ensure_ascii=False),
            )
        ),
    ]
    raw = model.invoke(messages)
    return raw if isinstance(raw, PreconsultDecision) else PreconsultDecision.model_validate(raw)


def _seed_special_population(
    slots: ConsultationSlots,
    state: Mapping[str, Any],
) -> ConsultationSlots:
    """把患者档案中的儿童、高龄或妊娠事实同步为特殊人群槽位。"""

    profile = state.get("patient_profile") or {}
    labels: list[str] = []
    age = profile.get("age")
    if isinstance(age, int) and age < 14:
        labels.append("儿童")
    elif isinstance(age, int) and age >= 65:
        labels.append("老年人")
    if profile.get("pregnancy") is True:
        labels.append("妊娠期")
    return merge_consultation_slots(slots, {"special_population": labels})


def _is_returning_from_retrieval(state: Mapping[str, Any]) -> bool:
    """判断当前是否是同一轮检索后的回跳，避免重复调用一次大模型。"""

    audit_log = state.get("audit_log", []) or []
    return bool(audit_log and audit_log[-1].get("node") == "graph_retrieval")


def preconsult_node(state: MedicalState) -> dict[str, Any]:
    """完成一轮预问诊决策，并返回需要合并进 LangGraph 的最小状态更新。

    执行顺序为：读取最新患者回答、规则/模型提取、增量合并槽位、判断上限与
    信息缺口、产生一次结构化检索请求或一个患者问题。检索时不增加追问计数；
    真正把问题展示给患者时才把 ``question_count`` 加一。
    """
    # 从状态取出旧槽位,如果没有槽位，就用空字典 {}
    old_slots = ConsultationSlots.model_validate(state.get("consultation_slots", {}))
    latest_text = latest_patient_text(state)
    # 更新旧槽位
    rule_updates = extract_rule_slot_updates(
        latest_text,
        current_question=state.get("current_question"),
        chief_complaint=str(state.get("chief_complaint", "")),
    )
    errors: list[dict[str, Any]] = []
    model_decision: PreconsultDecision | None = None

    # graph_retrieval 完成后会立即回到本节点；槽位在检索前已经提取过，此时再次请求模型既浪费调用，也可能对同一患者文本产生不一致结果。
    returning_from_retrieval = _is_returning_from_retrieval(state)
    if settings.llm_enabled and not returning_from_retrieval:
        try:
            model_decision = _invoke_model(state, latest_text)
        except Exception:
            errors.append(
                MedicalError(
                    category="llm",
                    code="preconsult_model_unavailable",
                    message="预问诊模型暂时不可用，已采用规则继续采集信息。",
                    retryable=True,
                    node="preconsult",
                ).model_dump()
            )

    # 先合并模型抽取结果
    slots = merge_consultation_slots(
        old_slots,
        model_decision.slot_updates if model_decision else {},
    )
    # 再合并规则结果,规则明确识内容具有更高优先级
    slots = merge_consultation_slots(slots, rule_updates)
    # 根据患者档案补充特殊人群的内容
    slots = _seed_special_population(slots, state)

    # 读取已经真正展示给患者的问题数
    rounds = max(0, int(state.get("question_count", 0)))
    max_rounds = max(0, int(state.get("max_question_count", settings.max_question_count)))
    # 建立包含新槽位的临时状态  '|'是 Python 的字典合并运算符，右侧同名字段覆盖左侧
    state_with_slots = dict(state) | {"consultation_slots": slots.model_dump()}
    # 明确下一个问题内容
    deterministic_question = next_missing_question(slots, state=state_with_slots)
    question = _select_question(
        deterministic_question,
        model_decision,
        previous_question=state.get("current_question"),
    )
    sufficient = _information_is_sufficient(slots, state_with_slots)
    # 模型可以建议结束，但代码仍以核心槽位是否充分为准；规则模式也使用同一标准。
    # 结束条件达到最大追问数；已没有缺失问题；核心信息已经充分。
    completed = rounds >= max_rounds or question is None or sufficient

    # 决定是否提出 RAG 请求
    retrieval = None if completed else _choose_retrieval_request(
        state_with_slots, slots, model_decision
    )
    # 根据检索元组更新retrieval_intent和retrieval_entities 并生成signature
    # 只有 _choose_retrieval_request 发现**"新的、还没查过的症状"**时才返回请求。大多数追问轮次根本不触发检索。
    need_retrieval = retrieval is not None
    retrieval_intent: RetrievalIntent | None = retrieval[0] if retrieval else None
    retrieval_entities = retrieval[1] if retrieval else []
    signature = (
        _retrieval_signature(retrieval_intent, retrieval_entities)
        if need_retrieval and retrieval_intent
        else None
    )
    # structured_llm：模型成功参与。
    # post_retrieval_rules：图谱检索后回跳，本次没有重复调用模型。
    # rule_fallback：模型关闭或不可用，使用规则。
    audit_entry: dict[str, Any] = {
        "node": "preconsult",
        "decision_source": (
            "structured_llm"
            if model_decision
            else ("post_retrieval_rules" if returning_from_retrieval else "rule_fallback")
        ),
        "slot_updates": sorted(
            set(rule_updates) | set(model_decision.slot_updates if model_decision else {})
        ),
        "completed": completed,
        "information_sufficient": sufficient,
        "question_field": _question_field(question),
    }
    if signature:
        audit_entry["retrieval_signature"] = signature
    # 生成节点状态更新
    update: dict[str, Any] = {
        "consultation_slots": slots.model_dump(),
        "current_question": None if completed or need_retrieval else question,
        "question_count": rounds if completed or need_retrieval else rounds + 1,
        "need_graph_retrieval": need_retrieval,
        "retrieval_intent": retrieval_intent,
        "retrieval_entities": retrieval_entities,
        "conversation_status": (
            "completed" if completed else ("preconsulting" if need_retrieval else "waiting_user")
        ),
        "audit_log": [audit_entry],
    }
    if errors:
        update["errors"] = errors
    return update
