---
title: Wiki Coverage Audit 2026-04-08
category: journal
tags:
  - journal
  - audit
  - wiki
  - coverage
sources:
  - workspace/wiki-coverage-audit-2026-04-08
created: 2026-04-08
updated: 2026-04-08
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

## 优先修复顺序

1. 先补 `reference -> entities` 缺口（当前最明显）。
2. 再补 `reference -> concepts/synthesis` 的剩余缺口。
3. 最后逐篇回补 `note -> reference/concepts/entities/synthesis` 入口段落。^[inferred]

## 关联页面

- [[notes/index]]
- [[references/index]]
- [[concepts/index]]
- [[entities/index]]
- [[synthesis/index]]
- [[log]]
