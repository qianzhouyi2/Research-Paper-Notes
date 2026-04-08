---
title: Papers Sources Deep Gap Audit 2026-04-08
category: journal
tags:
  - journal
  - audit
  - wiki-update
updated: 2026-04-08
summary: 对 papers_sources 的 9 篇论文做逐篇覆盖度盘点，定位 concepts、entities、synthesis 的细化缺口并给出补全目标。
---

# Papers Sources Deep Gap Audit 2026-04-08

## 盘点范围

- `papers_sources/Are aligned neural networks adversarially aligned 2306.15447`
- `papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation`
- `papers_sources/Hyena Hierarchy 2302.10866`
- `papers_sources/Maintaining Plasticity in Deep Continual Learning`
- `papers_sources/MeanSparse 2406.05927`
- `papers_sources/Research-Paper-Notes`（CIF、GPT-ST、Language Models Represent Space and Time、PanGu-π）

## 当前覆盖密度（盘点前）

- Are aligned neural networks adversarially aligned：`concept=1, entity=2, synthesis=1`
- Delving into Decision-based Black-box Attacks on Semantic Segmentation：`concept=5, entity=3, synthesis=2`
- Hyena Hierarchy：`concept=1, entity=1, synthesis=1`
- Maintaining Plasticity：`concept=2, entity=1, synthesis=1`
- MeanSparse：`concept=1, entity=0, synthesis=1`
- CIF：`concept=1, entity=1, synthesis=1`
- GPT-ST：`concept=1, entity=1, synthesis=1`
- Language Models Represent Space and Time：`concept=1, entity=1, synthesis=1`
- PanGu-π：`concept=1, entity=1, synthesis=1`

## 主要缺口

- 多数论文仍是“单锚点”链接，无法表达方法内部模块结构。
- `entities/` 在作者、模型、数据集层面明显偏稀疏，无法支持跨论文检索。
- `synthesis/` 主题覆盖偏粗，缺少“对齐鲁棒评估”“长上下文非注意力架构”“持续学习可塑性”等专页。
- 若不继续细化，`notes -> reference -> concept/entity/synthesis` 链接虽然完整，但知识图谱深度不足。

## 本轮补全目标

- 每篇论文至少补 2 个更细颗粒概念（模块级或评测协议级）。
- 每篇论文补 1-3 个关键实体（优先作者/模型/数据集）。
- 新增 3-4 个跨论文综合页，并回链到对应 reference 与 concept。
- 对 9 篇相关笔记的 `Wiki 关联` 做人工增强，替换自动最小链接。

## 补全后结果

- Are aligned neural networks adversarially aligned：`concept=3, entity=8, synthesis=2`
- Delving into Decision-based Black-box Attacks on Semantic Segmentation：`concept=5, entity=6, synthesis=2`
- Hyena Hierarchy：`concept=3, entity=3, synthesis=1`
- Maintaining Plasticity：`concept=4, entity=2, synthesis=1`
- MeanSparse：`concept=1, entity=3, synthesis=2`
- CIF：`concept=2, entity=2, synthesis=1`
- GPT-ST：`concept=3, entity=3, synthesis=2`
- Language Models Represent Space and Time：`concept=2, entity=2, synthesis=2`
- PanGu-π：`concept=3, entity=2, synthesis=2`

## 本轮新增资产

- 新增概念页：12
- 新增实体页：18
- 新增综合页：4
- 新增审计页：1
