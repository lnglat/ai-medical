// 真实三库初始化与正式数据导入：离线医学数据清洗、实体对齐与索引构建 数据契约中八类节点的稳定 ID 唯一约束。
// 请在 NEO4J_DATABASE 指定的数据库执行；重复执行不会重复创建约束。

CREATE CONSTRAINT disease_id_unique IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT symptom_id_unique IF NOT EXISTS FOR (n:Symptom) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT department_id_unique IF NOT EXISTS FOR (n:Department) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT check_id_unique IF NOT EXISTS FOR (n:Check) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT drug_id_unique IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT food_id_unique IF NOT EXISTS FOR (n:Food) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT cause_id_unique IF NOT EXISTS FOR (n:Cause) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT people_id_unique IF NOT EXISTS FOR (n:People) REQUIRE n.id IS UNIQUE;
