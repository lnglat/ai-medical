"""离线实体收集、审核映射复用和向量聚类。

嵌入器通过协议注入，单元测试可使用假向量，不依赖网络或大型模型。生产命令
默认加载本地 BGE；只有明确使用 ``--no-embeddings`` 时才降级为 exact 对齐。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from src.datasync.cleaning import clean_text, stable_id
from src.datasync.schemas import CleanMedicalRecord, EntityMappingRecord, EntityType


class TextEmbedder(Protocol):
    """最小嵌入接口；BGE 和测试假实现都只需提供 encode。"""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class LocalBgeEmbedder:
    """延迟加载本地 BGE，并固定使用 CPU 与归一化向量。"""

    def __init__(self, model_path: Path, *, batch_size: int = 128) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self._model = None
        self.dimension: int | None = None

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """首次调用才加载模型，输出普通列表方便序列化和测试。"""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                str(self.model_path), device="cpu", local_files_only=True
            )
        vectors = self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.dimension = int(vectors.shape[1]) if len(vectors) else None
        return vectors


FIELD_ENTITY_TYPES: dict[str, EntityType] = {
    "name": "disease",
    "symptoms": "symptom",
    "departments": "department",
    "checks": "check",
    "drugs": "drug",
    "foods_recommended": "food",
    "foods_avoided": "food",
    "causes": "cause",
    "people": "people",
}


def collect_entities(records: Iterable[CleanMedicalRecord]) -> dict[EntityType, Counter[str]]:
    """按实体类型统计词频，为稳定选择标准词提供依据。"""

    counters: dict[EntityType, Counter[str]] = {
        # 实体set()去重并生成空计数器
        entity_type: Counter() for entity_type in sorted(set(FIELD_ENTITY_TYPES.values()))
    }
    for record in records:
        for field, entity_type in FIELD_ENTITY_TYPES.items():
            value = getattr(record, field)
            # value是列表时直接使用，否则包装成列表
            values = [value] if isinstance(value, str) else value
            counters[entity_type].update(text for text in values if text)
    return counters  # {"disease": Counter({"高血压": 1, "糖尿病": 1}),...}


def _lsh_signatures(matrix: np.ndarray, *, bands: int = 8, bits_per_band: int = 8) -> list[tuple[int, ...]]:
    """为大实体集生成确定性的余弦 LSH 签名，避免构造完整 n×n 相似度矩阵。"""

    if not len(matrix):
        return []
    # 创建一个固定种子的随机数生成器
    rng = np.random.default_rng(20260819 + matrix.shape[1])
    # 生成 总平面数 × 维度 的随机高斯矩阵。
    planes = rng.standard_normal((bands * bits_per_band, matrix.shape[1]))\
    # 形成n×64 的布尔矩阵（每个词对每个平面取符号）
    signs = (matrix @ planes.T) >= 0
    # 把二进制位转成整数签名
    signatures: list[tuple[int, ...]] = []
    weights = 1 << np.arange(bits_per_band)
    # reshape(n, 8, 8)：把每个词的 64 位拆成 8 行 × 8 位
    for row in signs.reshape(len(matrix), bands, bits_per_band):
        # 每行 8 位按 weights = [1, 2, 4, 8, 16, 32, 64, 128] 加权求和 → 一个 0–255 的整数
        signatures.append(tuple(int(part @ weights) for part in row))
    # 返回一个长度为 n 的列表，每个元素是一个 8 元组，表示该词的 LSH 签名
    return signatures


def _cluster_assignments(
    terms: Sequence[str],
    vectors: Sequence[Sequence[float]],
    frequencies: Counter[str],
    *,
    cluster_threshold: float,
    review_threshold: float,
    exact_limit: int = 2000,
) -> dict[int, tuple[int, float, bool, str | None]]:
    """把每个词直接匹配到一个标准词代表，杜绝传递式链状误合并。

    小数据集精确比较所有标准词代表；大数据集用固定随机种子的余弦 LSH 缩小
    候选范围。无论采用哪种候选方式，最终合并都必须再次通过真实余弦阈值。
    返回值为 ``原词下标 -> (标准词下标, 置信度, 是否待审核, 审核候选)``。
    """
    # 向量必须是二维矩阵，且“一个词对应一个向量”
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(terms):
        raise ValueError("嵌入结果数量或维度与实体列表不一致") 
    # 按行给每个词向量算出长度，并以 (n,1) 的二维形状返回
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("嵌入模型返回了零向量，无法进行余弦聚类")
    # 归一化向量，方便后续余弦相似度计算
    matrix = matrix / norms

    priority = sorted(
        range(len(terms)),
        key=lambda index: (
            -frequencies[terms[index]], # 频率
            -sum(character.isalnum() for character in terms[index]), # 字母数字数
            terms[index], # Unicode 字典序兜底
        ),
    )
    signatures = _lsh_signatures(matrix) if len(terms) > exact_limit else []
    buckets: dict[tuple[int, int], set[int]] = {}
    representatives: list[int] = []
    assignments: dict[int, tuple[int, float, bool, str | None]] = {}

    # priority 是处理顺序——按词频降序、字母数字数降序、Unicode 字典序排好序的下标列表。
    for index in priority:
        if signatures: # 大数据集
            # 收集候选标准词下标。用 set 去重——因为一个词可能在多个 band 都命中同一个标准词
            candidates: set[int] = set()
            # signatures[index] 是一个 (int, int, ...) 8 元组，每带一个 0–255 的整数
            for band, signature in enumerate(signatures[index]):
                candidates.update(buckets.get((band, signature), set()))
            candidate_indexes = sorted(candidates)
        else: #  小数据集
            candidate_indexes = representatives

        #通过候选列表计算当前词向量与候选标准词向量的余弦相似度，找出最相似的标准词
        best_index: int | None = None
        best_similarity = -1.0
        if candidate_indexes:
            # 计算当前词向量与候选标准词向量的余弦相似度
            similarities = matrix[candidate_indexes] @ matrix[index]
            # 返回的在候选列表里与当前词最相似的标准词的相对位置
            position = int(np.argmax(similarities))
            # 返回的是priority排序后的候选标准词的下标，而不是原始terms列表的下标
            best_index = candidate_indexes[position]
            # 返回的最相似标准词的余弦相似度
            best_similarity = float(similarities[position])

        # 如果最相似的标准词的相似度达到自动合并阈值，则直接合并，不需要审核
        if best_index is not None and best_similarity >= cluster_threshold:
            assignments[index] = (best_index, best_similarity, False, None)
            continue

        # 是否待审核
        needs_review = best_index is not None and best_similarity >= review_threshold
        # 审核候选词文本
        candidate_text = terms[best_index] if needs_review and best_index is not None else None

        # 将 < cluster_threshold的标准词写入赋值记录
        assignments[index] = (index, 1.0 if not needs_review else best_similarity, needs_review, candidate_text)
        # 加入更新候选词列表
        representatives.append(index)

        # 生成 buckets
        if signatures:
            for band, signature in enumerate(signatures[index]):
                # 键不存在 → 插入空集合并返回它；键已存在 → 直接返回现有集合
                buckets.setdefault((band, signature), set()).add(index)

    # 低置信度候选是成对关系。后出现的词被标记后，也把作为候选的独立标准词
    # 标记为待审核，避免审核清单只展示关系的一侧。
    for index, (standard_index, confidence, needs_review, candidate) in list(assignments.items()):
        if not needs_review or standard_index != index or candidate is None:
            continue
        candidate_index = terms.index(candidate)
        candidate_assignment = assignments[candidate_index]
        if candidate_assignment[0] == candidate_index and not candidate_assignment[2]:
            assignments[candidate_index] = (
                candidate_index, confidence, True, terms[index]
            )
    return assignments


def align_entities(
    # 由 collect_entities() 生成
    counters: Mapping[EntityType, Counter[str]],
    *,
    # 键是 (实体类型, 原词)，值是标准词。
    reviewed_mapping: Mapping[tuple[EntityType, str], str] | None = None,
    embedder: TextEmbedder | None = None,
    cluster_threshold: float = 0.88,
    review_threshold: float = 0.80,
) -> list[EntityMappingRecord]:
    """生成完整实体映射。

    已审核 MySQL 映射拥有最高优先级。未命中词可经 BGE 聚类；低于自动合并阈值
    的相似项绝不强制合并，只有处于审核区间的项会标记 ``needs_review``。
    """
    # reviewed_mapping 是 load_reviewed_mysql_mapping() 查出来的,来源是 MySQL 的 entity_mapping 表
    # 且 SQL 里写了 WHERE review_status = 1。
    reviewed_mapping = reviewed_mapping or {}
    result: list[EntityMappingRecord] = []
    for entity_type in sorted(counters): # counters:{"disease": Counter({"高血压": 1, "糖尿病": 1}),...}
        frequencies = counters[entity_type]
        # 收集"没有有效审核映射、需要走聚类"的词
        pending: list[str] = []
        for term in sorted(frequencies):
            # 键是 (entity_type, 原词),值是对应的审核标准词
            reviewed = reviewed_mapping.get((entity_type, term))
            if reviewed:
                reviewed = clean_text(reviewed)
                if not reviewed:
                    pending.append(term)
                    continue
                result.append(EntityMappingRecord(
                    entity_id=stable_id(entity_type, reviewed), entity_type=entity_type,
                    original_text=term, standard_text=reviewed, source="mysql:entity_mapping",
                    method="reviewed_mysql", confidence=1.0, needs_review=False,
                    rationale="命中 review_status=1 的历史人工审核映射",
                ))
            else:
                pending.append(term)
        if not pending:
            continue

        if embedder: # 有嵌入器时，先生成向量再聚类
            vectors = embedder.encode(pending)
            assignments = _cluster_assignments(
                pending, vectors, frequencies,
                cluster_threshold=cluster_threshold, review_threshold=review_threshold,
            )
        else: # 没有嵌入器时，直接按原词作为标准词
            assignments = {index: (index, 1.0, False, None) for index in range(len(pending))}

        for index, term in enumerate(pending):
            # 解包四元组(标准词下标, 置信度, 是否待审核, 审核候选词)
            standard_index, confidence, needs_review, candidate = assignments[index]
            # 直接使用标准词下标从 pending 列表中取出标准词
            standard = pending[standard_index]
            if standard_index != index:
                method = "embedding_cluster"
                rationale = (
                    f"与标准词直接余弦相似度达到 {cluster_threshold:.2f}；"
                    "标准词按词频、信息完整度、Unicode 顺序选择"
                )
            elif needs_review:
                method = "low_confidence_candidate"
                rationale = f"与“{candidate}”相似但未达到自动合并阈值，保留独立并待审核"
            else:
                method = "exact"
                rationale = "未发现达到自动合并阈值的同义候选，保留独立实体"
            result.append(EntityMappingRecord(
                entity_id=stable_id(entity_type, standard), entity_type=entity_type,
                original_text=term, standard_text=standard,
                source="local_bge" if embedder else "local_exact", method=method,
                confidence=round(confidence, 6), needs_review=needs_review,
                rationale=rationale,
            ))
    return sorted(result, key=lambda item: (item.entity_type, item.original_text))


def mapping_lookup(records: Iterable[EntityMappingRecord]) -> dict[tuple[EntityType, str], str]:
    """将可追踪映射转成构建图谱时便于查询的字典。"""

    return {(item.entity_type, item.original_text): item.standard_text for item in records}
