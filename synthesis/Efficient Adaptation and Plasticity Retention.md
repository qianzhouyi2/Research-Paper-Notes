---
title: Efficient Adaptation and Plasticity Retention
category: synthesis
tags:
  - synthesis
  - adaptation
  - efficiency
  - plasticity
sources:
  - notes/LoRA Low-Rank Adaptation of Large Language Models.md
  - notes/Maintaining Plasticity in Deep Continual Learning.md
  - notes/PanGu-pi Nonlinearity Compensation.md
created: 2026-04-08
updated: 2026-04-10
summary: 该主题整合低秩微调、持续可塑性维护与结构补偿三类路线，关注在受限计算与参数预算下如何保持模型继续学习和继续适配的能力。
provenance:
  extracted: 0.75
  inferred: 0.23
  ambiguous: 0.02
---

# Efficient Adaptation and Plasticity Retention

## 综合结论

- 低成本适配不仅是减少参数量，也包括在长时程训练中避免能力僵化。
- 参数化增量、结构补偿和选择性重初始化分别从不同层面维持模型可更新性。^[inferred]

## 代表页面

- [[references/LoRA Low-Rank Adaptation of Large Language Models]]
- [[references/Maintaining Plasticity in Deep Continual Learning]]
- [[references/PanGu-pi Nonlinearity Compensation]]

## 关联概念

- [[concepts/Parameter-Efficient Fine-Tuning for LLMs]]
- [[concepts/Low-Rank Adaptation for LLMs]]
- [[concepts/Continual Backpropagation]]
- [[concepts/Loss of Plasticity in Continual Learning]]
- [[concepts/LLM Nonlinearity Compensation]]

## 联网补充

- LoRA 证明“冻结主干 + 低秩增量”可以显著降低适配开销，同时保持任务性能，是参数高效适配的基线范式。
- 与 Maintaining Plasticity、PanGu-π 的结论合并后可见：长期可持续适配需要同时处理参数效率、可塑性维护与结构非线性表达能力。
