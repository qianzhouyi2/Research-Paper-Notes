---
title: Probabilistic Inference-Time Control for LLMs
category: synthesis
tags:
  - synthesis
  - llm
  - inference
  - probabilistic
sources:
  - notes/Amortizing intractable inference in large language models.md
  - notes/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity.md
  - notes/Synergy-of-Thoughts Eliciting Efficient Reasoning in Hybrid Language Models.md
created: 2026-04-08
updated: 2026-04-08
summary: 该主题整合“后验采样、复杂度路由、双过程策略”三类方法，强调在推理阶段动态控制计算与不确定性。
provenance:
  extracted: 0.75
  inferred: 0.22
  ambiguous: 0.03
---

# Probabilistic Inference-Time Control for LLMs

## 综合结论

- 推理控制正在从固定解码参数走向“难度感知 + 概率后验采样 + 策略切换”联合机制。
- 目标不是单点提速，而是在预算内稳定提升复杂样本质量。^[inferred]

## 代表页面

- [[references/Amortizing intractable inference in large language models]]
- [[references/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity]]
- [[references/Synergy-of-Thoughts Eliciting Efficient Reasoning in Hybrid Language Models]]

## 关联概念

- [[concepts/Amortized Bayesian Inference for LLMs]]
- [[concepts/GFlowNet Posterior Sampling for Text Generation]]
- [[concepts/Success-Rate Based Complexity Labeling]]
- [[concepts/Task Complexity-Aware Inference Budgeting]]
- [[concepts/Adaptive Compute Routing]]
