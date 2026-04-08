---
title: Task Complexity-Aware Inference Budgeting
category: concept
tags:
  - concept
  - llm
  - inference
  - efficiency
sources:
  - notes/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity.md
  - notes/Synergy-of-Thoughts Eliciting Efficient Reasoning in Hybrid Language Models.md
  - notes/Amortizing intractable inference in large language models.md
created: 2026-04-08
updated: 2026-04-08
summary: 根据样本难度动态分配推理计算预算，以降低平均计算成本并保持关键样本性能。
provenance:
  extracted: 0.79
  inferred: 0.19
  ambiguous: 0.02
---

# Task Complexity-Aware Inference Budgeting

该范式强调“先估计难度，再决定推理深度”，本质是把固定预算推理改为按样本自适应计算。

## 关联页面

- [[references/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity]]
- [[references/Synergy-of-Thoughts Eliciting Efficient Reasoning in Hybrid Language Models]]
- [[references/Amortizing intractable inference in large language models]]
- [[synthesis/LLM Inference Efficiency and Scaling]]
