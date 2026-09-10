-- 仅用于已有 entity_mapping 表：把二态审核字段升级为三态审核状态。
-- 执行后重新运行 scripts/prepare_data.py --apply，自动映射会按离线产物回填为 0 或 2；
-- 原 review_status=1 的人工审核映射会被导入逻辑保留。

ALTER TABLE `ai_medical`.`entity_mapping`
  CHANGE COLUMN `is_reviewed` `review_status` TINYINT UNSIGNED NOT NULL DEFAULT 0
  COMMENT '0=待人工审核（低置信），1=已人工审核，2=高置信自动映射';
