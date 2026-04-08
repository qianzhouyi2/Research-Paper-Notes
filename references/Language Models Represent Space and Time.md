---
title: Language Models Represent Space and Time
category: reference
tags:
  - paper
  - llm
  - representation
  - spatio-temporal
sources:
  - notes/Language Models Represent Space and Time.md
created: 2026-04-08
updated: 2026-04-08
summary: 该工作通过系统探针证据支持 LLM 内部存在跨实体统一且可线性解码的时空表示。
provenance:
  extracted: 0.87
  inferred: 0.11
  ambiguous: 0.02
---

# Language Models Represent Space and Time

## 摘要卡

- 原始笔记：[[notes/Language Models Represent Space and Time]]
- 核心结论：这项工作用系统探针证据支持了“LLM 内部存在跨实体统一、可线性解码的时空表示”。

## 细化方法锚点

- 层级激活提取：比较不同层对时空变量的可解码性。
- 线性探针评估：联合 `R^2`、Spearman 与接近误差验证稳定性。
- 泛化协议：跨提示、跨实体、分块留出评估排除记忆查表效应。^[inferred]

## 关联页面

- [[concepts/Spatio-Temporal Pretraining for Language Models]]
- [[concepts/Linear Probe World-Model Evaluation]]
- [[entities/Wes Gurnee]]
- [[entities/Max Tegmark]]
- [[references/GPT-ST Spatio-Temporal Pretraining]]
- [[synthesis/Spatio-Temporal Representation in Language Models]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]
