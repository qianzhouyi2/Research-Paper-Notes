---
title: Success-Rate Based Complexity Labeling
category: concept
tags:
  - concept
  - llm
  - routing
  - evaluation
sources:
  - notes/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity.md
created: 2026-04-08
updated: 2026-04-08
summary: 通过多模型多次采样成功率定义任务复杂度标签，降低单次评测噪声对路由学习的影响。
provenance:
  extracted: 0.83
  inferred: 0.15
  ambiguous: 0.02
---

# Success-Rate Based Complexity Labeling

核心思想是先估计任务在模型族上的可解性分布，再据此定义复杂度等级。

## 关联页面

- [[references/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity]]
- [[concepts/Task Complexity-Aware Inference Budgeting]]
- [[synthesis/Probabilistic Inference-Time Control for LLMs]]

