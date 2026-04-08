---
title: Amortized Bayesian Inference for LLMs
category: concept
tags:
  - concept
  - llm
  - inference
  - bayesian
sources:
  - notes/Amortizing intractable inference in large language models.md
created: 2026-04-08
updated: 2026-04-08
summary: 将难解后验推理转化为可复用的近似策略学习，降低 LLM 推理时的采样与约束搜索成本。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# Amortized Bayesian Inference for LLMs

该范式通过离线学习“近似后验策略”，在在线推理阶段复用，避免每次都做昂贵的显式后验推断。

## 关联页面

- [[references/Amortizing intractable inference in large language models]]
- [[concepts/Task Complexity-Aware Inference Budgeting]]
- [[synthesis/Probabilistic Inference-Time Control for LLMs]]

