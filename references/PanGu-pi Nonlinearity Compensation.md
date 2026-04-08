---
title: PanGu-pi Nonlinearity Compensation
category: reference
tags:
  - paper
  - llm
  - efficiency
  - architecture
sources:
  - notes/PanGu-pi Nonlinearity Compensation.md
created: 2026-04-08
updated: 2026-04-08
summary: PanGu-pi 通过增强捷径与级联激活在较低增量成本下改善大模型非线性表达与推理效率。
provenance:
  extracted: 0.85
  inferred: 0.13
  ambiguous: 0.02
---

# PanGu-pi Nonlinearity Compensation

## 摘要卡

- 原始笔记：[[notes/PanGu-pi Nonlinearity Compensation]]
- 核心结论：PanGu-π 通过“增强捷径 + 级联激活”在低增量成本下提升了 LLM 的非线性表达能力、效率与领域实用性。

## 细化方法锚点

- 增强捷径：在注意力块并行引入补偿支路以缓解特征坍塌。
- 级联激活：在 FFN 中增加可学习激活补偿，提高非线性容量。
- 组合验证：在通用模型与领域模型上验证收益稳定性。

## 关联页面

- [[concepts/LLM Nonlinearity Compensation]]
- [[concepts/Augmented Shortcut for Attention Blocks]]
- [[concepts/Cascaded Activation Nonlinearity Compensation]]
- [[entities/Yunhe Wang]]
- [[entities/Hanting Chen]]
- [[synthesis/LLM Inference Efficiency and Scaling]]
- [[synthesis/Long-Context Architecture Without Full Attention]]
- [[references/LoRA Low-Rank Adaptation of Large Language Models]]
