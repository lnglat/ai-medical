"""离线医学数据清洗、实体对齐与索引构建 离线医学数据整理包。

这里的代码只由 ``scripts/prepare_data.py`` 或测试调用，不会进入患者在线问诊链路。
"""

from src.datasync.pipeline import PipelineConfig, PipelineResult, run_pipeline

__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline"]
