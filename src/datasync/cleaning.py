"""原始 JSONL 检查、文本清洗和字段规范化。"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from src.datasync.schemas import CleanMedicalRecord, InputProfile, RawMedicalRecord, RejectedRecord


HTML_TAG = re.compile(r"<[^>]*>")
WHITESPACE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    """清洗字符串，同时保留有意义的中文标点。

    NFKC 会统一全角/半角及兼容字符；先反转义 HTML 再删除标签，可避免
    ``&lt;b&gt;`` 这种编码标签残留。字典、嵌套列表、数字等异常类型不会被
    强行 ``str()`` 成垃圾实体，而是抛出错误并由上层写入拒绝记录。
    """

    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"文本字段必须是字符串或 null，实际为 {type(value).__name__}")
    value = html.unescape(value) # 反转义 HTML 实体
    value = HTML_TAG.sub(" ", value) #删除 <p>、</p> 标签
    value = unicodedata.normalize("NFKC", value) # 统一全角/半角及兼容字符
    return WHITESPACE.sub(" ", value).strip() # 多个空白压成一个


def unique_text_list(value: Any) -> list[str]:
    """将字符串或列表统一为去空、去重且保持首次出现顺序的列表。"""

    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise TypeError(f"多值字段必须是字符串、列表或 null，实际为 {type(value).__name__}")
    result: list[str] = []

    seen: set[str] = set()
    for item in values:
        # seen 是集合，用于快速判断是否出现过
        text = clean_text(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def stable_id(prefix: str, text: str, *, length: int = 20) -> str:
    """依据“类型 + 规范化文本”生成确定性 ID，而不是使用随机 UUID。"""

    normalized = clean_text(text).casefold() # 所有字母都变成小写
    digest = hashlib.sha256(f"{prefix}\0{normalized}".encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def inspect_jsonl(path: Path) -> InputProfile:
    """只读扫描 JSONL 的编码、字段、空值和坏行数量。"""

    digest = hashlib.sha256()
    fields: dict[str, int] = {}
    empty_fields: dict[str, int] = {}
    total = blank = valid = invalid = 0
    with path.open("rb") as binary:
        for raw_line in binary:
            digest.update(raw_line)
            total += 1
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                invalid += 1
                continue
            if not line.strip():
                blank += 1
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("顶层必须是dict")
                # json.JSONDecodeError：专门针对 JSON 解析失败
                # ValueError：针对值类型转换错误。
            except (json.JSONDecodeError, ValueError):
                invalid += 1
                continue
            valid += 1
            for key, item in value.items():
                fields[key] = fields.get(key, 0) + 1
                if item is None or item == "" or item == []:
                    empty_fields[key] = empty_fields.get(key, 0) + 1
    empty_ratios = {
        key: round(empty_fields.get(key, 0) / count, 6) if count else 0.0
        for key, count in fields.items()
    }
    return InputProfile(
        path=str(path), sha256=digest.hexdigest(), total_lines=total, blank_lines=blank,
        valid_json_records=valid, invalid_json_records=invalid,
        field_counts=dict(sorted(fields.items())), empty_field_counts=dict(sorted(empty_fields.items())),
        empty_field_ratios=dict(sorted(empty_ratios.items())),
    )


def clean_jsonl(path: Path, *, limit: int | None = None) -> tuple[list[CleanMedicalRecord], list[RejectedRecord]]:
    """逐行清洗知识图谱输入；坏行进入拒绝列表而非静默丢弃。"""

    cleaned: list[CleanMedicalRecord] = []
    rejected: list[RejectedRecord] = []
    source = path.as_posix() #避免 Windows 反斜杠路径在 JSON 中被转义
    # 使用二进制逐行读取，确保单行 UTF-8 损坏时只拒绝该行，而不是由文本迭代器在进入 try 之前抛错并中断整条流水线。
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if limit is not None and len(cleaned) + len(rejected) >= limit:
                break
            raw_for_error: Any = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            try:
                line = raw_line.decode("utf-8")
                payload = json.loads(line)
                raw_for_error = payload
                raw = RawMedicalRecord.model_validate(payload)
                name = clean_text(raw.name)
                if not name:
                    raise ValueError("缺少必填疾病名 name")
                # 行号和规范化 JSON 同时进入摘要：同名但内容不同的原始行不会再共用 ID，
                # 同一文件内容重复构建时仍得到完全相同的结果。
                # dumps() 将字典转成 JSON 字符串
                canonical_payload = json.dumps(
                    # ensure_ascii=False：保留非英文字符原样
                    # separators=(",", ":")：定义 JSON 的分隔符，去掉多余空格，保证相同内容的 JSON 字符串一致
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                record = CleanMedicalRecord(
                    record_id=stable_id("record", f"{line_number}|{canonical_payload}"), source_file=source,
                    source_line=line_number, name=name, desc=clean_text(raw.desc),
                    symptoms=unique_text_list(raw.symptom),
                    departments=unique_text_list(raw.department),
                    checks=unique_text_list(raw.check), drugs=unique_text_list(raw.drug),
                    foods_recommended=unique_text_list(raw.eat),
                    foods_avoided=unique_text_list(raw.not_eat),
                    causes=unique_text_list(raw.cause), people=unique_text_list(raw.people),
                )
                cleaned.append(record)
            except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                rejected.append(RejectedRecord(
                    source_file=source, source_line=line_number,
                    reason=f"{type(exc).__name__}: {exc}", raw_record=raw_for_error,
                ))
    return cleaned, rejected
