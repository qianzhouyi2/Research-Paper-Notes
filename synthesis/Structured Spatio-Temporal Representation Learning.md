---
title: Structured Spatio-Temporal Representation Learning
category: synthesis
tags:
  - synthesis
  - spatio-temporal
  - pretraining
  - representation
sources:
  - papers_sources/Research-Paper-Notes/CIF.md
  - papers_sources/Research-Paper-Notes/GPT-ST预训练时空框架.md
  - papers_sources/Research-Paper-Notes/LLM体现出时空概念.md
created: 2026-04-08
updated: 2026-04-08
summary: 时空表示学习可分为“结构化学习机制”和“可解码性验证机制”两条线，前者提升迁移，后者验证内部表征真实性。
provenance:
  extracted: 0.74
  inferred: 0.24
  ambiguous: 0.02
---

# Structured Spatio-Temporal Representation Learning

## 两条证据链

- 学习机制：CIF 的单调对齐、GPT-ST 的聚类感知预训练与分层超图编码。
- 验证机制：线性探针评估 LLM 内部时空表示是否可泛化解码。

## 代表论文

- [[references/CIF Continuous Integrate-and-Fire]]
- [[references/GPT-ST Spatio-Temporal Pretraining]]
- [[references/Language Models Represent Space and Time]]

## 关联概念

- [[concepts/Continuous Integrate-and-Fire Alignment]]
- [[concepts/Quantity-Aware Emission Control in CIF]]
- [[concepts/Cluster-Aware Masked Pretraining]]
- [[concepts/Hierarchical Spatio-Temporal Hypergraph Encoding]]
- [[concepts/Linear Probe World-Model Evaluation]]

