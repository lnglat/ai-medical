"""摘要 Agent：把患者事实整理成可追溯的摘要和病历草稿。

本模块只消费患者原始主诉、预问诊槽位和分诊结果，不读取数据库，也不决定工作流
下一跳。摘要完全由经过 Pydantic 校验的状态事实确定性生成，不调用大模型；这样既
没有额外网络延迟和 Token 成本，也不会因为模型改写而混入无来源医疗事实。
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from src.graph.state import MedicalState
from src.models.schemas import (
    ConsultationSlots,
    DifferentialDirection,
    MedicalRecordDraft,
    PossibleEvaluation,
    PreconsultSummary,
    TriageResult,
)


SAFETY_NOTICE = "本摘要仅用于预问诊信息整理，不构成诊断或治疗建议。"
DIFFERENTIAL_NOTICE = (
    "以下内容来自症状与医学知识图谱的关联，仅供预问诊信息整理，"
    "不代表诊断或检查医嘱。"
)
MAX_DIRECTIONS = 3
MAX_EVALUATIONS_PER_DIRECTION = 3

_FIELD_LABELS: dict[str, str] = {
    "symptom": "症状",
    "onset": "起病方式",
    "duration": "病程",
    "severity": "严重程度",
    "location": "部位",
    "characteristics": "症状性质",
    "aggravating_or_relieving_factors": "诱发或缓解因素",
    "accompanying_symptoms": "伴随症状",
    "medical_history": "既往史",
    "medication_history": "用药史",
    "allergy_history": "过敏史",
    "special_population": "特殊人群信息",
}

_NEGATIVE_PREFIXES = ("否认", "没有", "未出现", "不伴", "并无")

# “无力”“无尿”等本身是阳性症状，不能仅因首字为“无”就判为否定。
_POSITIVE_WU_TERMS = ("无力", "无尿", "无汗", "无痛性")
_SYMPTOM_ALIASES = {
    "发烧": "发热",
    "头疼": "头痛",
    "喘不上气": "气促",
    "呼吸急促": "气促",
    "呼吸困难": "气促",
}


def _canonical_symptom(value: str) -> str:
    text = value.strip()
    return _SYMPTOM_ALIASES.get(text, text)


class SummaryBundle(BaseModel):
    """把确定性生成的摘要和病历草稿封装为一个内部结构化结果。"""
    # 禁止内部调用方传入未声明字段，保证容器只包含摘要和病历草稿。
    model_config = ConfigDict(extra="forbid")

    summary: PreconsultSummary
    medical_record_draft: MedicalRecordDraft


def _clean_items(values: list[str]) -> list[str]:
    """清除空白并保持患者陈述的原始顺序，避免摘要重复展示同一事实。"""

    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


def _is_negative(value: str) -> bool:
    """preconsult_agent 已明确保存的否定陈述；未知值不能被当成阴性。"""

    text = value.strip()
    if text.startswith(_POSITIVE_WU_TERMS):
        return False
    return text == "无" or text.startswith((*_NEGATIVE_PREFIXES, "无"))


def _split_findings(slots: ConsultationSlots) -> tuple[list[str], list[str]]:
    """将槽位事实拆分为阳性与阴性发现，同时保留患者原始措辞。"""

    positives: list[str] = []
    negatives: list[str] = []
    # * 是列表展开
    for value in [*slots.symptom, *slots.accompanying_symptoms]:
        (negatives if _is_negative(value) else positives).append(value)
    for value in [*slots.medical_history, *slots.medication_history, *slots.allergy_history]:
        if _is_negative(value):
            negatives.append(value)
    return _clean_items(positives), _clean_items(negatives)


def _missing_information(slots: ConsultationSlots) -> list[str]:
    """返回尚未采集的中文字段名；空值表示未知，而不是患者否认。"""

    return [_FIELD_LABELS[name] for name, value in slots.model_dump().items() if not value]


def _present_illness(chief_complaint: str, slots: ConsultationSlots) -> str:
    """用患者原始主诉开头，再追加已有结构化事实，不采用图谱推测。"""

    details: list[str] = []
    scalar_fields = (
        ("起病方式", slots.onset),
        ("病程", slots.duration),
        ("严重程度", slots.severity),
        ("部位", slots.location),
        ("性质", slots.characteristics),
        ("诱发或缓解因素", slots.aggravating_or_relieving_factors),
    )
    details.extend(f"{label}：{value}" for label, value in scalar_fields if value)
    if slots.accompanying_symptoms:
        details.append(f"伴随症状：{'、'.join(_clean_items(slots.accompanying_symptoms))}")
    if slots.special_population:
        details.append(f"特殊人群信息：{'、'.join(_clean_items(slots.special_population))}")

    text = f"患者原话：{chief_complaint or '未提供'}"
    if details:
        text += "；" + "；".join(details)
    return text + "。"


def _standardized_symptom_entities(state: Mapping[str, Any]) -> list[str]:
    """读取明确标为症状的标准实体，仅用于结构化辅助备注。

    只采用图谱证据的来源实体，因为它通常对应上游提交的患者症状；目标实体可能是
    图谱扩展出的关联疾病、科室或其他症状，不能被当成患者已经陈述的事实。缺少实体
    类型时宁可不展示，也不根据名称猜测。
    """

    entities = [
        str(item.get("source_entity", "")).strip()
        for item in state.get("retrieved_evidence", []) or []
        # 函数只接受"symptom",只读取source_entity,不读取 target_entity
        if item.get("source_entity_type") == "symptom"
        and str(item.get("source_entity", "")).strip()
    ]
    return _clean_items(entities)


def _negative_symptom_name(value: str) -> str | None:
    """从具体阴性事实中取症状名；泛化否认不能参与疾病冲突计算。"""

    text = value.strip()
    for prefix in _NEGATIVE_PREFIXES:
        if text.startswith(prefix):
            symptom = text[len(prefix):].strip()
            if symptom and symptom not in {
                "其他伴随症状", "相关既往病史", "近期及长期用药", "已知过敏史",
            }:
                return symptom
    return None


def _patient_symptom_sets(
    state: Mapping[str, Any],
    slots: ConsultationSlots,
) -> tuple[set[str], set[str]]:
    """汇总多轮阳性/阴性症状，并吸收检索产生的阳性标准实体名。"""

    positives = {
        _canonical_symptom(value)
        for value in [*slots.symptom, *slots.accompanying_symptoms]
        if value.strip() and not _is_negative(value)
    }
    negatives = {
        _canonical_symptom(symptom)
        for value in [*slots.symptom, *slots.accompanying_symptoms]
        if (symptom := _negative_symptom_name(value)) is not None
    }
    positives.update(
        _canonical_symptom(value)
        for value in _standardized_symptom_entities(state)
        if _canonical_symptom(value) not in negatives
    )
    positives.difference_update(negatives)
    return positives, negatives


def _differential_outputs(
    state: Mapping[str, Any],
    slots: ConsultationSlots,
) -> tuple[list[DifferentialDirection], list[PossibleEvaluation]]:
    """按可解释规则筛选疾病方向，并只绑定这些方向的图谱检查。"""

    positives, negatives = _patient_symptom_sets(state, slots)
    disease_symptoms: dict[str, set[str]] = {}
    disease_checks: dict[str, list[Mapping[str, Any]]] = {}

    for item in state.get("retrieved_evidence", []) or []:
        source = str(item.get("source_entity", "")).strip()
        target = str(item.get("target_entity", "")).strip()
        source_type = item.get("source_entity_type")
        target_type = item.get("target_entity_type")
        relation = item.get("relation")
        if relation == "HAS_SYMPTOM" and source_type == "symptom" and target_type == "disease":
            disease_symptoms.setdefault(target, set()).add(_canonical_symptom(source))
        elif relation == "HAS_SYMPTOM" and source_type == "disease" and target_type == "symptom":
            disease_symptoms.setdefault(source, set()).add(_canonical_symptom(target))
        elif relation == "RECOMMENDS_CHECK" and source_type == "disease" and target_type == "check":
            disease_checks.setdefault(source, []).append(item)

    ranked: list[tuple[int, int, int, str, list[str], list[str]]] = []
    for disease, related_symptoms in disease_symptoms.items():
        supporting = sorted(related_symptoms & positives)
        if len(supporting) < 2:
            continue
        conflicting = sorted(related_symptoms & negatives)
        support_count = len(supporting)
        conflict_count = len(conflicting)
        ranked.append((
            support_count - conflict_count,
            support_count,
            conflict_count,
            disease,
            supporting,
            conflicting,
        ))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))

    directions = [
        DifferentialDirection(
            disease_name=disease,
            supporting_symptoms=supporting,
            conflicting_negative_symptoms=conflicting,
            support_count=support_count,
            conflict_count=conflict_count,
            support_score=support_score,
            notice=DIFFERENTIAL_NOTICE,
        )
        for support_score, support_count, conflict_count, disease, supporting, conflicting
        in ranked[:MAX_DIRECTIONS]
    ]

    evaluations: list[PossibleEvaluation] = []
    for direction in directions:
        seen_checks: set[str] = set()
        for item in disease_checks.get(direction.disease_name, []):
            check_name = str(item.get("target_entity", "")).strip()
            if not check_name or check_name in seen_checks:
                continue
            seen_checks.add(check_name)
            evaluations.append(PossibleEvaluation(
                check_name=check_name,
                disease_direction=direction.disease_name,
                evidence_source=str(item.get("evidence_source") or "neo4j"),
                source_records=_clean_items([
                    str(record) for record in item.get("source_records", []) if str(record).strip()
                ]),
            ))
            if len(seen_checks) >= MAX_EVALUATIONS_PER_DIRECTION:
                break
    return directions, evaluations


def _fact_source_paths() -> dict[str, list[str]]:
    """声明摘要关键字段的可信来源路径，供审计而不重复保存患者敏感文本。"""

    return {
        "summary.chief_complaint": ["state.chief_complaint"],
        "summary.present_illness_summary": [
            "state.chief_complaint",
            "state.consultation_slots",
        ],
        "summary.key_positive_findings": ["state.consultation_slots"],
        "summary.key_negative_findings": ["state.consultation_slots"],
        "summary.missing_information": ["state.consultation_slots"],
        "summary.triage_recommendation": ["state.triage_result"],
        "summary.differential_directions": [
            "state.consultation_slots",
            "state.retrieved_evidence[relation=HAS_SYMPTOM]",
        ],
        "summary.possible_evaluations": [
            "summary.differential_directions",
            "state.retrieved_evidence[relation=RECOMMENDS_CHECK]",
        ],
        "medical_record_draft": [
            "state.chief_complaint",
            "state.consultation_slots",
            "state.triage_result",
        ],
        "medical_record_draft.note.standardized_symptom_entities": [
            "state.retrieved_evidence[*].source_entity[type=symptom]",
        ],
    }


def build_template_bundle(
    state: Mapping[str, Any],
    slots: ConsultationSlots,
    triage: TriageResult,
) -> SummaryBundle:
    """从经过校验的状态构建确定性摘要和病历草稿。

    输入中的图谱证据不会成为患者事实；它只可能在上游协助提问。这样即使图谱返回
    某疾病名称，摘要也不会把关联关系写成患者诊断。
    """

    chief_complaint = str(state.get("chief_complaint", "")).strip() or "未提供"
    positives, negatives = _split_findings(slots)
    triage_recommendation = "、".join(_clean_items(triage.recommended_departments)) or "待确认"
    # 生成现病史
    present_illness = _present_illness(chief_complaint, slots)
    # 整理用药、过敏和标准化症状
    medication = "、".join(_clean_items(slots.medication_history)) or "未提供"
    allergy = "、".join(_clean_items(slots.allergy_history)) or "未提供"
    standardized_symptoms = _standardized_symptom_entities(state)
    differential_directions, possible_evaluations = _differential_outputs(state, slots)
    note = SAFETY_NOTICE
    if standardized_symptoms:
        note += (
            f" 结构化辅助症状词：{'、'.join(standardized_symptoms)}；"
            "来源于知识检索，仅供信息整理参考，不代表诊断。"
        )
    # 预问诊摘要
    summary = PreconsultSummary(
        chief_complaint=chief_complaint,
        present_illness_summary=present_illness,
        key_positive_findings=positives,
        key_negative_findings=negatives,
        missing_information=_missing_information(slots),
        triage_recommendation=triage_recommendation,
        safety_notice=SAFETY_NOTICE,
        differential_directions=differential_directions,
        possible_evaluations=possible_evaluations,
    )
    # 病历草稿
    draft = MedicalRecordDraft(
        chief_complaint=chief_complaint,
        history_of_present_illness=present_illness,
        past_history="、".join(_clean_items(slots.medical_history)) or "未提供",
        medication_and_allergy_history=f"用药：{medication}；过敏：{allergy}",
        preliminary_department=triage_recommendation,
        note=note,
    )
    return SummaryBundle(summary=summary, medical_record_draft=draft)


def _is_emergency_state(state: Mapping[str, Any]) -> bool:
    """防御性识别紧急分支，确保误调用摘要节点时也不生成普通病历。"""

    triage = state.get("triage_result") or {}
    # 只要任意条件成立，就返回 True
    return (
        state.get("conversation_status") == "emergency_ended"
        or triage.get("urgency") == "emergency"
        or bool(triage.get("red_flags"))
        # 缺少字段时 .get() 返回 None。None 不应自动被当成“明确禁止继续”
        or triage.get("should_continue_preconsult") is False
    )


def summary_node(state: MedicalState) -> dict[str, Any]:
    """生成摘要与病历草稿，并只返回需要合并进 LangGraph 的字段。

    正常路径只使用患者原话、结构化槽位和分诊结果构建输出。紧急状态只追加审计
    记录，不写 ``summary`` 或 ``medical_record_draft``。
    """

    if _is_emergency_state(state):
        return {
            "conversation_status": "emergency_ended",
            "current_question": None,  # 不再继续追问
            "audit_log": [{"node": "summary", "skipped": True, "reason": "emergency"}],
        }

    slots = ConsultationSlots.model_validate(state.get("consultation_slots", {}))
    triage = TriageResult.model_validate(state.get("triage_result"))
    # 只从已经校验的状态构建结果，不再发起额外模型请求。
    result = build_template_bundle(state, slots, triage)

    update: dict[str, Any] = {
        "summary": result.summary.model_dump(),
        "medical_record_draft": result.medical_record_draft.model_dump(),
        "conversation_status": "completed",
        "current_question": None,
        "audit_log": [{
            "node": "summary",
            "decision_source": "deterministic_template",
            "selected_output_trusted": True,
            "graph_evidence_used_as_patient_fact": False,
            "standardized_symptom_entity_count": len(_standardized_symptom_entities(state)),
            "differential_direction_count": len(result.summary.differential_directions),
            "possible_evaluation_count": len(result.summary.possible_evaluations),
            "fact_source_paths": _fact_source_paths(),
        }],
    }
    return update
