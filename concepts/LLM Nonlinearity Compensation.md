---
title: LLM Nonlinearity Compensation
category: concept
tags:
  - concept
  - llm
  - architecture
  - efficiency
sources:
  - notes/PanGu-pi Nonlinearity Compensation.md
created: 2026-04-08
updated: 2026-04-08
summary: 非线性补偿通过结构增量与激活设计增强 LLM 表达能力，在较低额外成本下提升推理与任务表现。
provenance:
  extracted: 0.87
  inferred: 0.11
  ambiguous: 0.02
---

# LLM Nonlinearity Compensation

该路线重点解决“规模增长下非线性表达不足”问题。

## 联网补充

- PanGu-π 的论点不是“再堆参数”，而是深层 Transformer 会出现特征相似化与秩退化，需要在注意力捷径和 FFN 激活两侧补非线性。
- 这属于架构级改造而非推理时技巧，收益要靠训练期吸收，因此更接近“更好的 backbone”而不是“更便宜的调用策略”。

## 关联页面

- [[references/PanGu-pi Nonlinearity Compensation]]
- [[concepts/Augmented Shortcut for Attention Blocks]]
- [[concepts/Cascaded Activation Nonlinearity Compensation]]
- [[synthesis/LLM Inference Efficiency and Scaling]]
- [[synthesis/Long-Context Architecture Without Full Attention]]

