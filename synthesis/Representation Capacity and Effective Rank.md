---
title: Representation Capacity and Effective Rank
category: synthesis
tags:
  - synthesis
  - representation
  - capacity
  - diagnostics
sources:
  - notes/Maintaining Plasticity in Deep Continual Learning.md
  - notes/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature.md
  - notes/PanGu-pi Nonlinearity Compensation.md
  - notes/Language Models Represent Space and Time.md
created: 2026-04-08
updated: 2026-04-10
summary: 该主题从有效秩、特征多样性、线性可解码结构等视角审视模型表示容量，强调表示退化往往先于性能退化显现。
provenance:
  extracted: 0.74
  inferred: 0.24
  ambiguous: 0.02
---

# Representation Capacity and Effective Rank

## 综合结论

- 表示容量不足通常表现为有效秩下降、特征坍塌或可解码结构受损。
- 结构补偿、稀疏化后处理和持续重初始化都可被视为“恢复表示可分性”的不同干预。^[inferred]

## 代表页面

- [[references/Maintaining Plasticity in Deep Continual Learning]]
- [[references/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification]]
- [[references/PanGu-pi Nonlinearity Compensation]]
- [[references/Language Models Represent Space and Time]]

## 关联概念

- [[concepts/Feature Effective Rank Diagnostics]]
- [[concepts/Mean-Centered Feature Sparsification]]
- [[concepts/Proximal L0 Feature Sparsification]]
- [[concepts/LLM Nonlinearity Compensation]]
- [[concepts/Linear Probe World-Model Evaluation]]

## 联网补充

- 持续学习与鲁棒增强工作共同表明：有效秩、特征稀疏性与表征多样性会先于任务精度变化，适合作为早期退化信号。
- 《Language Models Represent Space and Time》提供了线性探针证据，说明“可解码结构”可以作为容量与结构化表示质量的补充指标。

## ?????2026-04-10?

- ?????????????????????????????
- ?????? - 表示容量不足通常表现为有效秩下降、特征坍塌或可解码结构受损。 - 结构补偿、稀疏化后处理和持续重初始化都可被视为“恢复表示可分性”的不同干预。^[inferred]
- ????????????
- ??????????? URL???????????????^[ambiguous]
- ????????????arXiv / DOI / ???????????????????
- ?????????????????????

