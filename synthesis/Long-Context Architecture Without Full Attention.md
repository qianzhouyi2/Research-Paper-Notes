---
title: Long-Context Architecture Without Full Attention
category: synthesis
tags:
  - synthesis
  - long-context
  - llm
  - architecture
sources:
  - papers_sources/Hyena Hierarchy 2302.10866/3_hyena.tex
  - papers_sources/Research-Paper-Notes/PanGu-π.md
  - notes/Hyena Hierarchy Towards Larger Convolutional Language Models.md
  - notes/PanGu-pi Nonlinearity Compensation.md
created: 2026-04-08
updated: 2026-04-10
summary: 长上下文架构正从“全注意力”转向“次二次全局混合+非线性补偿”组合路线，以换取更高上下文效率比。
provenance:
  extracted: 0.69
  inferred: 0.28
  ambiguous: 0.03
---

# Long-Context Architecture Without Full Attention

## 共同方向

- Hyena 路线：用隐式长卷积和数据控制门控替代全注意力的全局混合。
- PanGu-π 路线：在 Transformer 结构内用捷径与激活补偿增强深层表达。

## 代表论文

- [[references/Hyena Hierarchy Towards Larger Convolutional Language Models]]
- [[references/PanGu-pi Nonlinearity Compensation]]
- [[references/LoRA Low-Rank Adaptation of Large Language Models]]

## 关联概念

- [[concepts/Hyena Operator]]
- [[concepts/Data-Controlled Gating in Sequence Models]]
- [[concepts/Implicit Long Convolution Parameterization]]
- [[concepts/Augmented Shortcut for Attention Blocks]]
- [[concepts/Cascaded Activation Nonlinearity Compensation]]

## 联网补充

- Hyena Hierarchy 展示了隐式长卷积与门控组合可实现次二次复杂度的长序列建模，并在多项设置下接近或匹配 Transformer 表现。
- PanGu-π 的非线性补偿结果提示：当弱化全注意力时，需要用结构补偿机制维持表达能力与训练稳定性。
