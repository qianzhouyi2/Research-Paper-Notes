---
title: Spatio-Temporal Representation in Language Models
category: synthesis
tags:
  - synthesis
  - llm
  - spatio-temporal
  - representation
sources:
  - notes/GPT-ST Spatio-Temporal Pretraining.md
  - notes/Language Models Represent Space and Time.md
created: 2026-04-08
updated: 2026-04-10
summary: 时空主题下形成“预训练增强 + 表征探针验证”两条互补证据链，分别回答如何学到时空结构与是否真的学到时空结构。
provenance:
  extracted: 0.8
  inferred: 0.18
  ambiguous: 0.02
---

# Spatio-Temporal Representation in Language Models

## 双线证据

- 训练侧：[[references/GPT-ST Spatio-Temporal Pretraining]] 提供结构化预训练机制。
- 解释侧：[[references/Language Models Represent Space and Time]] 提供内部表示可解码证据。

## 关联页面

- [[concepts/Spatio-Temporal Pretraining for Language Models]]
- [[concepts/Linear Probe World-Model Evaluation]]
- [[concepts/Cluster-Aware Masked Pretraining]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]

## 联网补充

- GPT-ST 通过生成式时空预训练提升下游时空预测任务，说明结构化时空先验对泛化有效。
- 《Language Models Represent Space and Time》显示主流语言模型可线性解码出空间与时间关系，提供“模型已学到何种时空结构”的证据链。
