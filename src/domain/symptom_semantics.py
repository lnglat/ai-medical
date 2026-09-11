"""共享的症状别名标准化与中文否定解析。

槽位保留患者原始措辞；本模块只为匹配、去重、检索和方向计算提供统一语义，
避免预问诊、方向筛选和摘要对同一句症状产生不同判断。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SymptomPolarity = Literal["positive", "negative", "unknown"]

SYMPTOM_ALIASES: dict[str, str] = {
    "发烧": "发热",
    "头疼": "头痛",
    "喘不上气": "气促",
    "呼吸急促": "气促",
    "呼吸困难": "气促",
}

# 长前缀必须排在短前缀前，避免“没有气促”被先按“没”切成“有气促”。
NEGATIVE_PREFIXES = (
    "未出现", "不存在", "没有", "否认", "不伴", "并无", "未见", "不是", "没", "无",
)
POSITIVE_WU_TERMS = ("无力", "无尿", "无汗", "无痛性")
GENERIC_NEGATIVE_TARGETS = {
    "其他伴随症状", "相关既往病史", "近期及长期用药", "已知过敏史",
}

# 单字“无”只有在后面不是固定阳性医学词时才作为否定线索。
_NEGATION_CUE_PATTERN = re.compile(
    r"(?:未出现|不存在|没有|否认|不伴|并无|未见|不是|没|无(?!力|尿|汗|痛性))"
)
_CLAUSE_SEPARATORS = ("，", ",", "。", "；", ";", "但")


@dataclass(frozen=True)
class ParsedSymptom:
    """一条患者症状表达对应的标准名称与极性。"""

    original_text: str
    name: str | None
    polarity: SymptomPolarity


def canonicalize_symptom(value: str) -> str:
    """把已知口语别名转换为稳定症状名，未知名称原样保留。"""

    text = value.strip()
    return SYMPTOM_ALIASES.get(text, text)


def parse_symptom(value: str) -> ParsedSymptom:
    """解析槽位级症状表达；不把“无力”等阳性医学词误判为否定。"""

    text = value.strip()
    if not text:
        return ParsedSymptom(value, None, "unknown")
    if text.startswith(POSITIVE_WU_TERMS):
        return ParsedSymptom(value, canonicalize_symptom(text), "positive")
    if text == "无":
        return ParsedSymptom(value, None, "negative")
    for prefix in NEGATIVE_PREFIXES:
        if text.startswith(prefix):
            target = text[len(prefix):].strip()
            name = canonicalize_symptom(target) if target else None
            if name in GENERIC_NEGATIVE_TARGETS:
                name = None
            return ParsedSymptom(value, name, "negative")
    return ParsedSymptom(value, canonicalize_symptom(text), "positive")


def is_negative_statement(value: str) -> bool:
    """判断已经进入槽位的一条陈述是否为明确阴性。"""

    return parse_symptom(value).polarity == "negative"


def has_negation_cue(text: str) -> bool:
    """判断自由文本是否包含否定线索，同时保护“无力”等阳性词。"""

    return bool(_NEGATION_CUE_PATTERN.search(text))


def term_is_negated(text: str, term: str, *, window: int = 8) -> bool:
    """按局部短句判断指定症状是否被否定，以最后一次陈述为准。"""

    position = text.rfind(term)
    if position < 0:
        return False
    clause_start = max(
        (text.rfind(separator, 0, position) for separator in _CLAUSE_SEPARATORS),
        default=-1,
    )
    prefix = text[clause_start + 1:position]
    return has_negation_cue(prefix[-window:])
