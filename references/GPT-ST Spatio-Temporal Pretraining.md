---
title: GPT-ST Spatio-Temporal Pretraining
category: reference
tags:
  - paper
  - llm
  - spatio-temporal
  - pretraining
sources:
  - notes/GPT-ST Spatio-Temporal Pretraining.md
created: 2026-04-08
updated: 2026-04-08
summary: GPT-ST 通过结构感知预训练学习可迁移时空关系，在多任务上带来稳定下游性能增益。
provenance:
  extracted: 0.87
  inferred: 0.11
  ambiguous: 0.02
---

# GPT-ST Spatio-Temporal Pretraining

## 摘要卡

- 原始笔记：[[notes/GPT-ST Spatio-Temporal Pretraining]]
- 核心结论：GPT-ST 用结构感知预训练机制把时空关系学得更可迁移，并在多任务上稳定提升下游预测性能。

## 细化方法锚点

- 时间模式编码：时间动态与区域个体参数联合调制表示。
- 分层空间建模：簇内结构与簇间迁移关系协同学习。
- 掩码策略：聚类感知掩码优于纯随机掩码。

## 关联页面

- [[concepts/Spatio-Temporal Pretraining for Language Models]]
- [[concepts/Cluster-Aware Masked Pretraining]]
- [[concepts/Hierarchical Spatio-Temporal Hypergraph Encoding]]
- [[entities/GPT-ST]]
- [[entities/Peiyi Wang]]
- [[entities/Li Dong]]
- [[references/Language Models Represent Space and Time]]
- [[synthesis/Spatio-Temporal Representation in Language Models]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]
