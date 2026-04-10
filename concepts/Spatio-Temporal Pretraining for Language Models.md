---
title: Spatio-Temporal Pretraining for Language Models
category: concept
tags:
  - concept
  - llm
  - spatio-temporal
  - pretraining
sources:
  - notes/GPT-ST Spatio-Temporal Pretraining.md
  - notes/Language Models Represent Space and Time.md
created: 2026-04-08
updated: 2026-04-10
summary: 时空预训练通过结构化目标让语言模型更稳定地学习空间与时间关系，并提升相关下游任务迁移能力。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Spatio-Temporal Pretraining for Language Models

通过在预训练阶段显式引入时空结构信号，提升模型时空表示的可解码性与泛化性。

## 联网补充

- GPT-ST 说明可以在预训练阶段显式注入时空归纳偏置，而“Language Models Represent Space and Time”又表明这类结构后来能在线性探针中被读出。
- 这个概念适合真正依赖时间顺序和空间拓扑的数据；如果时空只是弱元数据，额外结构化预训练不一定有同等收益。

## 关联页面

- [[references/GPT-ST Spatio-Temporal Pretraining]]
- [[references/Language Models Represent Space and Time]]
- [[concepts/Cluster-Aware Masked Pretraining]]
- [[concepts/Hierarchical Spatio-Temporal Hypergraph Encoding]]
- [[concepts/Linear Probe World-Model Evaluation]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]
