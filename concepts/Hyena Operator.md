---
title: Hyena Operator
category: concept
tags:
  - concept
  - long-context
  - convolution
  - sequence-modeling
sources:
  - papers_sources/Hyena Hierarchy 2302.10866/main.tex
  - papers_sources/2302.10866.tar
created: 2026-04-08
updated: 2026-04-10
summary: Hyena Operator 以长卷积与数据控制门控交织递归，构建可替代注意力的次二次序列混合算子，面向超长上下文更具扩展性。
provenance:
  extracted: 0.89
  inferred: 0.1
  ambiguous: 0.01
---

# Hyena Operator

## 定义

Hyena 是一种次二次序列算子，核心由“隐式长卷积 + 数据控制门控 + 递归交织”构成，用于替代标准注意力层。

## 关键特征

- 能在超长序列下保持全局混合能力。
- 通过隐式滤波器参数化降低长卷积参数负担。
- 计算复杂度随序列长度增长更慢，长上下文优势更明显。

## 设计启发

- 先把“高效全局混合”与“输入条件控制”拆开，再在层内递归融合。
- 该路径可与注意力形成可插拔混合架构。^[inferred]

## 联网补充

- PMLR 2023 正文把 Hyena 定义为 attention 的 subquadratic drop-in replacement，由 implicitly parametrized long convolutions 与 data-controlled gating 交织构成。
- 论文报告其长上下文优势会随长度放大，在 8k 序列上约 2 倍、64k 上约 100 倍快于优化注意力，因此它首先是长上下文架构杠杆。

## 关联页面

- [[references/Hyena Hierarchy Towards Larger Convolutional Language Models]]
- [[notes/Hyena Hierarchy Towards Larger Convolutional Language Models]]
- [[concepts/Data-Controlled Gating in Sequence Models]]
- [[concepts/Implicit Long Convolution Parameterization]]
- [[synthesis/Long-Context Architecture Without Full Attention]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Hyena Operator category: concept tags: - concept - long-context - convolution - sequence-modeling sources: - papers_sources/Hyena Hierarchy 2302.10866/main.tex - papers_sources/2302.10866.tar created: 2026-04-08 updated: 2026-04-09 summary: Hyena O...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

