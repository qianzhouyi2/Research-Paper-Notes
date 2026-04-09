---
title: Wiki Coverage Audit 2026-04-08
category: journal
tags:
  - journal
  - audit
  - wiki
sources:
  - workspace/wiki-coverage-audit-2026-04-08
created: 2026-04-08
updated: 2026-04-09
summary: 本页记录 2026-04-08 的全量缺口盘点结果，覆盖 notes 到 references 的映射完整性及 concepts/entities/synthesis 链接覆盖度。
provenance:
  extracted: 0.91
  inferred: 0.08
  ambiguous: 0.01
---

# Wiki Coverage Audit 2026-04-08

## 首轮盘点结果

- `notes` 总数（排除索引页）：25
- `references` 映射缺口：0（全部 note 已有对应 reference）
- reference 层覆盖缺口：
  - 无 `concepts` 链接：5
  - 无 `entities` 链接：16
  - 无 `synthesis` 链接：3
- note 正文覆盖缺口（目前仍以笔记原文为主）：
  - 无 `concepts` 链接：24
  - 无 `entities` 链接：24
  - 无 `synthesis` 链接：24

## 图谱健康度

- 关键知识库页孤岛数（`references/concepts/entities/synthesis`）：0
- 说明：结构层已有互链，但 note 层到知识层链接仍薄弱。^[inferred]

## 二轮补齐后复测

- `notes` 映射缺口：0
- `reference` 层缺口：
  - 无 `concepts` 链接：0
  - 无 `entities` 链接：0
  - 无 `synthesis` 链接：0
- `notes` 正文缺口：
  - 无 `references` 链接：0
  - 无 `concepts` 链接：0
  - 无 `entities` 链接：0
  - 无 `synthesis` 链接：0
- 关键知识库页孤岛数：0

## 三轮细扫后复测（notes 语义补链）

- `notes` 总数（排除索引页）：25
- `notes` 关联锚点下限：
  - `concepts`：每篇至少 2 个
  - `entities`：每篇至少 2 个
  - `synthesis`：每篇至少 2 个
- `notes` 分布（细扫后）：
  - `concepts >= 3`：17 篇
  - `entities >= 3`：10 篇
  - `synthesis >= 2`：25 篇
- 新增结构页（本轮）：5 个 `concepts`、3 个 `entities`、3 个 `synthesis`

## 四轮细扫后复测（notes 深层语义补链）

- `notes` 总数（排除索引页）：25
- `notes` 关联锚点下限：
  - `concepts`：每篇至少 2 个
  - `entities`：每篇至少 2 个
  - `synthesis`：每篇至少 2 个
- `notes` 分布（四轮后）：
  - `concepts >= 3`：21 篇
  - `entities >= 3`：12 篇
  - `synthesis >= 3`：3 篇
- 新增结构页（本轮）：6 个 `concepts`、4 个 `entities`、1 个 `synthesis`

## 五轮细扫后复测（notes 精修补链）

- `notes` 总数（排除索引页）：25
- `notes` 关联锚点下限：
  - `concepts`：每篇至少 3 个
  - `entities`：每篇至少 3 个
  - `synthesis`：每篇至少 2 个
- `notes` 分布（五轮后）：
  - `concepts >= 3`：25 篇
  - `entities >= 3`：25 篇
  - `synthesis >= 3`：9 篇
- 新增结构页（本轮）：4 个 `concepts`、12 个 `entities`、2 个 `synthesis`

## 六轮细扫后复测（notes 综合收口）

- `notes` 总数（排除索引页）：25
- `notes` 关联锚点下限：
  - `concepts`：每篇至少 3 个
  - `entities`：每篇至少 3 个
  - `synthesis`：每篇至少 3 个
- `notes` 分布（六轮后）：
  - `concepts >= 3`：25 篇
  - `entities >= 3`：25 篇
  - `synthesis >= 3`：25 篇
- 新增结构页（本轮）：2 个 `synthesis`

## 七轮联网核验后复测（facts / metadata refresh）

- `notes` 总数（排除索引页）：25
- `references` 总数：26
- 结构覆盖保持不变：
  - `concepts >= 3`：25 篇
  - `entities >= 3`：25 篇
  - `synthesis >= 3`：25 篇
- 联网纠错与收口结果：
  - 修正 6 篇 notes 的作者 / venue / source metadata
  - 修正 26 个 `references` 的联网核验说明，其中后半轮补齐剩余 20 个 reference 卡
  - 新增 10 个校验后作者实体页，移除 7 个误归因实体页
  - 4 个 `references` 补充现有作者实体锚点
- 结构层复检：
  - `concepts` 未发现需基于联网事实修正的漂移项
  - `synthesis` 未发现需基于联网事实修正的漂移项

## 八轮联网核验后复测（concept refresh）

- `concepts` 总数：50
- `concepts` 联网复核覆盖：50/50
- 本轮实质更新：
  - 27 个轻量 concept 页新增 `联网补充`
  - 重点补足机制定义、适用边界与相邻概念区分
  - 未新增或删除 structure pages
- 结构总量保持不变：
  - `notes`：26
  - `references`：26
  - `concepts`：50
  - `entities`：67
  - `synthesis`：21
- 复核结论：
  - `concepts` 层已从“定义卡”提升到“定义 + 联网补充”的双层结构
  - richer concept 页未发现需要按一手来源重写的事实漂移

## 九轮联网核验后复测（concept refresh 2）

- `concepts` 总数：50
- 第二轮 concept 联网补充覆盖：23/23 剩余页面
- 本轮实质更新：
  - 22 个学术 concept 页新增 `联网补充`
  - 1 个本地 workflow 页补充“不适用论文型联网核验”的边界说明
  - 重点收口 segmentation attacks、adversarial alignment、Hyena、plasticity、GPT-ST、CIF 等簇
- 累计结果：
  - 50/50 `concepts` 页面现已带有 `联网补充`
  - `notes`：26
  - `references`：26
  - `entities`：67
  - `synthesis`：21
- 复核结论：
  - `concepts` 层已从“部分联网补充”闭合到“全量联网补充”
  - 本轮未新增结构页，主要是把厚概念卡补齐到与轻量概念卡一致的解释粒度

## 十轮联网核验后复测（entity refresh）

- `entities` 总数：67
- `entities` 联网补充覆盖：67/67
- 类型分布：
  - `author`：52
  - `model`：9
  - `dataset`：5
  - `benchmark`：1
- 本轮实质更新：
  - 所有实体页新增 `联网补充`
  - 重点补足作者归因、模型官方定义、数据集范围与 benchmark 角色
  - 未新增或删除结构页
- 结构总量保持不变：
  - `notes`：26
  - `references`：26
  - `concepts`：50
  - `entities`：67
  - `synthesis`：21
- 复核结论：
  - `entities` 层已从“命名占位”提升到“实体定义 + 本库角色”的双层结构
  - 这轮之后，`concepts` 与 `entities` 两层都已完成全量联网补充

## 十一轮联网核验后复测（synthesis refresh）

- `synthesis` 总数：21
- `synthesis` 联网补充覆盖：21/21
- 本轮实质更新：
  - 所有综合页新增 `联网补充`
  - 重点补足鲁棒评测、结构化推理、推理时编排、PEFT 与时空表示的外部事实锚点
  - 未新增或删除结构页
- 结构总量保持不变：
  - `notes`：26
  - `references`：26
  - `concepts`：50
  - `entities`：67
  - `synthesis`：21
- 复核结论：
  - `synthesis` 层已从“主题结论卡”提升到“主题结论 + 联网补充”双层结构
  - 至此 `concepts / entities / synthesis` 三层均完成全量联网补充

## 优先修复顺序

1. 把 `notes` 里的 `论文信息` 区统一成和 `references` 一样的联网核验格式。
2. 统一新入库页面的 `联网补充` 写法，避免层间粒度不一致（尤其是 reference 到 synthesis 的映射说明）。
3. 后续 ingest 新论文时优先复用现有实体页，避免重复创建近义 author/model/dataset 卡。^[inferred]

## 关联页面

- [[notes/index]]
- [[references/index]]
- [[concepts/index]]
- [[entities/index]]
- [[synthesis/index]]
- [[log]]
