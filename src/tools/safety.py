"""分诊阶段的确定性红旗症状规则。

本模块是分诊 Agent 前面的安全闸门：它只判断患者描述中是否出现需要立即线下
评估的风险信号，不诊断具体疾病。规则先处理常见中文否定表达，再结合年龄、
妊娠状态等基础资料判断特殊人群风险；一旦命中，后续大模型不能撤销结果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models.schemas import PatientProfile


@dataclass(frozen=True)
class RedFlagRule:
    """一条可解释的红旗规则。

    ``keywords`` 是能够独立触发该规则的患者表述。这里有意使用较具体的短语，
    避免把普通的“疼”“出血”等宽泛词语直接解释成紧急医学结论。
    """

    code: str  # 稳定的程序编号
    label: str  # 展示给业务代码和审计日志
    keywords: tuple[str, ...]  # 用于匹配患者主诉


@dataclass(frozen=True)
class RedFlagMatch:
    """红旗规则的一次命中，供分诊审计使用。"""

    code: str
    label: str
    matched_term: str


RED_FLAG_RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule(
        code="altered_consciousness",
        label="意识障碍",
        keywords=("昏迷", "意识不清", "意识障碍", "神志不清", "叫不醒"),
    ),
    RedFlagRule(
        code="acute_neurological_deficit",
        label="急性神经症状",
        keywords=(
            "突然口角歪斜", "突发口角歪斜", "突然说话不清", "突发说话不清",
            "一侧肢体无力", "单侧肢体无力", "突然偏瘫", "突发偏瘫",
        ),
    ),
    RedFlagRule(
        code="severe_chest_symptoms",
        label="严重胸部症状",
        keywords=(
            "剧烈胸痛", "压榨性胸痛", "胸痛伴大汗", "胸痛伴出汗",
            "胸痛向左臂放射", "胸痛向背部放射",
        ),
    ),
    RedFlagRule(
        code="major_bleeding",
        label="大出血",
        keywords=(
            "大出血", "大量出血", "出血不止", "止不住血", "大量呕血",
            "呕血不止", "大量咯血", "大量便血", "喷射性出血",
        ),
    ),
    RedFlagRule(
        code="severe_breathing_difficulty",
        label="严重呼吸困难",
        keywords=(
            "严重呼吸困难", "呼吸困难", "喘不上气", "无法呼吸", "不能呼吸",
            "嘴唇发紫", "口唇发紫",
        ),
    ),
)


'''
否定词和症状之间可以出现“任何、明显、严重、阴道”等限定词。例如匹配“否认严重呼吸困难”里的短词“呼吸困难”时，前缀是“否认严重”，
仍应识别为否定。允许有限长度的中文间隔，再排除“未缓解”和“无明显诱因后突发”等实际是在描述症状持续或发生方式的结构。
 '''
# (?: ... )	括号内的否定词会被匹配，但不会被单独保存
# (?P<gap> ... ）定义一个名叫 gap 的“小盒子”，把匹配到的内容单独保存
# $ 强制要求匹配的内容必须出现在字符串的末尾。
_NEGATION_WITH_GAP = re.compile(
    r"(?:没有|没|无|未见|未出现|否认|不伴|并无|不存在)(?P<gap>[\u4e00-\u9fff]{0,8})$"
)
_NON_NEGATING_GAP_TERMS = (
    "缓解", "改善", "好转", "减轻", "消失", "诱因", "想到", "现在", "目前",
    "后来", "随后", "反而",
)

_CLAUSE_SEPARATOR = re.compile(r"[，。；;！？!?、]|但是|但|不过|然而")

# 判断症状词前有无否定词
def _term_is_negated(text: str, start: int) -> bool:
    """判断指定症状词前面是否存在同一分句内的直接否定表达。"""

    # 取症状词之前的所有文本
    prefix = text[:start]
    # 只看当前分句末尾，防止上一句话中的否定词影响下一句话的阳性症状。
    clause = _CLAUSE_SEPARATOR.split(prefix)[-1]
    # re.sub(规则, 替换成什么, 要处理的文本)
    # 患者可能在否定词和症状间输入空格、制表符或换行，统一移除后再判断。
    clause = re.sub(r"\s+", "", clause)
    negation = _NEGATION_WITH_GAP.search(clause)
    if negation is None:
        return False
    gap = negation.group("gap")
    return not any(term in gap for term in _NON_NEGATING_GAP_TERMS)


def _find_positive_term(text: str, terms: tuple[str, ...]) -> str | None:
    """返回第一项未被否定的命中词；全部未出现或被否定时返回 ``None``。"""

    for term in terms:
        # re.escape() 把词里的正则特殊字符转义成字面量
        # IGNORECASE 忽略大小写
        # re.finditer 在 text 里找出 term 的所有出现位置,逐个返回 match
        for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
            if not _term_is_negated(text, match.start()):
                return match.group(0)
    return None


def assess_red_flag_details(
    chief_complaint: str,
    profile: PatientProfile,
) -> list[RedFlagMatch]:
    """返回主诉命中的可解释红旗规则明细。

    输出只描述风险规则命中，不包含疾病名称。年龄和妊娠规则必须与危险症状
    同时出现，避免仅因属于特殊人群就把普通不适一律判为急诊。
    """

    text = chief_complaint.strip()
    matches: list[RedFlagMatch] = []

    for rule in RED_FLAG_RULES:
        # 判断有无阳性症状词
        matched_term = _find_positive_term(text, rule.keywords)
        if matched_term is not None:
            matches.append(RedFlagMatch(rule.code, rule.label, matched_term))

    # 年龄按整岁存储，因此 age=0 代表不足一岁；合并明确高热表述时保守分流。
    if profile.age == 0:
        infant_fever_term = _find_positive_term(
            text, ("高热", "发烧到38", "发热到38", "体温38", "体温39", "体温40")
        )
        if infant_fever_term:
            matches.append(RedFlagMatch("young_infant_fever", "低龄婴儿高热", infant_fever_term))

    # 妊娠状态必须再同时出现危险表述才触发；简单的“怀孕”本身不是红旗。
    if profile.pregnancy:
        pregnancy_term = _find_positive_term(
            text, ("阴道大量出血", "阴道出血不止", "持续剧烈腹痛", "剧烈下腹痛")
        )
        if pregnancy_term:
            matches.append(RedFlagMatch("pregnancy_emergency", "妊娠期危险症状", pregnancy_term))

    return matches



# 只有于tests里测试
def assess_red_flags(chief_complaint: str, profile: PatientProfile) -> list[str]:
    """返回去重后的红旗类别，保持 LangGraph 多 Agent 主流程 已使用的简单工具接口。"""

    labels: list[str] = []
    for match in assess_red_flag_details(chief_complaint, profile):
        if match.label not in labels:
            labels.append(match.label)
    return labels
