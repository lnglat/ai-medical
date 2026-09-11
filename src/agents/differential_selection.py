"""确定性鉴别方向筛选：只消费患者槽位和症状图谱证据。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.domain.symptom_semantics import canonicalize_symptom, parse_symptom
from src.graph.state import MedicalState
from src.models.schemas import ConsultationSlots, DifferentialDirection


DIFFERENTIAL_NOTICE = (
    "以下内容来自症状与医学知识图谱的关联，仅供预问诊信息整理，"
    "不代表诊断或检查医嘱。"
)
MAX_DIRECTIONS = 3


def _evidence_dict(item: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    return item if isinstance(item, Mapping) else item.model_dump()


def _patient_symptom_sets(
    slots: ConsultationSlots,
    evidence: Sequence[Mapping[str, Any] | Any],
) -> tuple[set[str], set[str]]:
    values = [*slots.symptom, *slots.accompanying_symptoms]
    parsed = [parse_symptom(value) for value in values]
    positives = {
        item.name for item in parsed if item.polarity == "positive" and item.name
    }
    negatives = {
        item.name for item in parsed if item.polarity == "negative" and item.name
    }
    # 症状来源实体由上游知识工具标准化，可补充进入患者已确认的症状集合。
    positives.update(
        canonicalize_symptom(str(item.get("source_entity", "")))
        for raw in evidence
        # := 是 Python 的“海象运算符”：调用 _evidence_dict(raw)，并将结果同时赋值给 item
        if (item := _evidence_dict(raw)).get("source_entity_type") == "symptom"
        and str(item.get("source_entity", "")).strip()
        and canonicalize_symptom(str(item.get("source_entity", ""))) not in negatives
    )
    positives.difference_update(negatives)
    return positives, negatives


def select_differential_directions(
    consultation_slots: ConsultationSlots,
    evidence: Sequence[Mapping[str, Any] | Any],
) -> list[DifferentialDirection]:
    """按净支持分筛选至多三个方向，不调用模型或数据库。"""

    # 把图谱证据转换为“疾病 → 症状集合”
    positives, negatives = _patient_symptom_sets(consultation_slots, evidence)
    disease_symptoms: dict[str, set[str]] = {}
    for raw in evidence:
        item = _evidence_dict(raw)
        source = str(item.get("source_entity", "")).strip()
        target = str(item.get("target_entity", "")).strip()
        # 类型不完整的证据不会贡献疾病关联
        if item.get("relation") != "HAS_SYMPTOM" or not source or not target:
            continue
        # 同时支持两种关系方向(症状 → 疾病 或 疾病 → 症状)，但只保留疾病 → 关联症状集合
        if item.get("source_entity_type") == "symptom" and item.get("target_entity_type") == "disease":
            disease_symptoms.setdefault(target, set()).add(canonicalize_symptom(source))
        elif item.get("source_entity_type") == "disease" and item.get("target_entity_type") == "symptom":
            disease_symptoms.setdefault(source, set()).add(canonicalize_symptom(target))

    ranked: list[tuple[int, int, int, str, list[str], list[str]]] = []
    for disease, related_symptoms in disease_symptoms.items():
        supporting = sorted(related_symptoms & positives)
        if len(supporting) < 2:
            continue
        # related_symptoms：该疾病相关的全部症状集合。 negatives：患者明确没有的症状集合。 &：集合交集
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
    # 1. 净支持分高的优先
    # 2. 若净支持分相同，阳性支持症状更多的优先
    # 3. 若仍相同，冲突阴性症状更少的优先
    # 4. 若仍相同，疾病名称字典序靠前的优先
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    return [
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


def differential_selection_node(state: MedicalState) -> dict[str, Any]:
    """把鉴别证据转换为最终方向，后续检查只能以此结果为输入。"""

    slots = ConsultationSlots.model_validate(state.get("consultation_slots", {}))
    differential_evidence = list(state.get("differential_evidence", []) or [])
    # 如果 differential_evidence 没有提供专门用于鉴别诊断的证据，就从全部检索证据中拿“疾病—症状”关系作为替代证据
    if not differential_evidence:
        differential_evidence = [
            item for item in state.get("retrieved_evidence", []) or []
            if _evidence_dict(item).get("relation") == "HAS_SYMPTOM"
        ]
    directions = select_differential_directions(
        slots,
        differential_evidence,
    )
    return {
        "differential_directions": [item.model_dump() for item in directions],
        "conversation_status": "summarizing",
        "current_question": None,
        "audit_log": [{
            "node": "differential_selection",
            "decision_source": "deterministic_rules",
            "differential_evidence_count": len(differential_evidence),
            "differential_direction_count": len(directions),
        }],
    }
