---
title: Cluster-Aware Masked Pretraining
category: concept
tags:
  - concept
  - spatio-temporal
  - pretraining
  - graph-learning
sources:
  - papers_sources/Research-Paper-Notes/GPT-ST预训练时空框架.md
  - notes/GPT-ST Spatio-Temporal Pretraining.md
created: 2026-04-08
updated: 2026-04-08
summary: 聚类感知掩码把预训练目标与簇内簇间结构绑定，较纯随机掩码更能学习可迁移时空关系。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# Cluster-Aware Masked Pretraining

## 定义

把总掩码预算拆分为“结构感知掩码 + 随机掩码”，优先覆盖同簇元素并逐步引导跨簇关系学习。

## 价值

- 提升时空表示的结构一致性。
- 在多下游模型上更稳定迁移。^[inferred]

## 关联页面

- [[references/GPT-ST Spatio-Temporal Pretraining]]
- [[concepts/Spatio-Temporal Pretraining for Language Models]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]

