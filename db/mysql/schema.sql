-- 真实三库初始化与正式数据导入：ai-medical 项目 MySQL 幂等初始化脚本。
-- 只创建缺失对象，不包含 DROP、DELETE 或清空数据的语句。

CREATE DATABASE IF NOT EXISTS `ai_medical`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `ai_medical`.`entity_mapping` (
  `id` VARCHAR(64) NOT NULL COMMENT '离线医学数据清洗、实体对齐与索引构建 标准实体稳定 ID，同一 ID 可对应多个同义词',
  `synonym` TEXT NOT NULL COMMENT '待标准化的完整原始词',
  `std_name` TEXT NOT NULL COMMENT '完整标准实体名称',
  `entity_schema` VARCHAR(32) NOT NULL COMMENT '离线医学数据清洗、实体对齐与索引构建 entity_type',
  `synonym_hash` BINARY(32) NOT NULL COMMENT '完整 synonym 的 SHA-256，用于等价唯一键',
  `review_status` TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '0=待人工审核（低置信），1=已人工审核，2=高置信自动映射',
  PRIMARY KEY (`entity_schema`, `synonym_hash`),
  KEY `idx_entity_mapping_id` (`id`),
  KEY `idx_entity_mapping_lookup` (`entity_schema`, `synonym`(191))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
