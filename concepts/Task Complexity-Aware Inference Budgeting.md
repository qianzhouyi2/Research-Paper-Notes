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

## 联网补充

- 这一类方法的共同点是只在不确定或困难样本上加预算，预算形式可以是更大模型、更深搜索、更多 verifier 或更复杂后验采样。
- 真正的瓶颈不是有没有预算器，而是预算信号准不准；估计器一旦误判，系统只会多出调度复杂度却拿不到质量回报。

## 关联页面

- [[references/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity]]
- [[references/Synergy-of-Thoughts Eliciting Efficient Reasoning in Hybrid Language Models]]
- [[references/Amortizing intractable inference in large language models]]
- [[synthesis/LLM Inference Efficiency and Scaling]]

