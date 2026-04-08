---
title: Linear Probe World-Model Evaluation
category: concept
tags:
  - concept
  - llm
  - interpretability
  - probing
sources:
  - papers_sources/Research-Paper-Notes/LLM体现出时空概念.md
  - notes/Language Models Represent Space and Time.md
created: 2026-04-08
updated: 2026-04-08
summary: 线性探针世界模型评估通过层级激活回归时空变量，检验模型是否形成可解码且可泛化的内部结构表示。
provenance:
  extracted: 0.9
  inferred: 0.08
  ambiguous: 0.02
---

# Linear Probe World-Model Evaluation

## 定义

对不同层激活训练线性探针，预测地理坐标或时间变量，并结合泛化与鲁棒测试验证表示稳定性。

## 评价重点

- 线性可解码性是否跨提示、跨实体成立。
- 严格留出设定下性能是否维持。

## 关联页面

- [[references/Language Models Represent Space and Time]]
- [[entities/Max Tegmark]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]

