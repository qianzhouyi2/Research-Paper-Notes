---
title: Data-Controlled Gating in Sequence Models
category: concept
tags:
  - concept
  - long-context
  - sequence-modeling
  - convolution
sources:
  - papers_sources/Hyena Hierarchy 2302.10866/3_hyena.tex
  - notes/Hyena Hierarchy Towards Larger Convolutional Language Models.md
created: 2026-04-08
updated: 2026-04-08
summary: 数据控制门控通过输入依赖的逐步调制提升长卷积算子表达力，是 Hyena 的核心增益来源之一。
provenance:
  extracted: 0.87
  inferred: 0.11
  ambiguous: 0.02
---

# Data-Controlled Gating in Sequence Models

## 定义

在序列混合中使用输入相关门控系数，对卷积输出做逐步调制，形成“内容条件化”的全局混合算子。

## 工程意义

- 提升次二次算子表达能力。
- 在不回退到全注意力的前提下保留上下文选择性。^[inferred]

## 关联页面

- [[references/Hyena Hierarchy Towards Larger Convolutional Language Models]]
- [[concepts/Hyena Operator]]
- [[synthesis/Long-Context Architecture Without Full Attention]]

