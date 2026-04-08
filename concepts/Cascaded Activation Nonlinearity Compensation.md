---
title: Cascaded Activation Nonlinearity Compensation
category: concept
tags:
  - concept
  - llm
  - architecture
  - nonlinearity
sources:
  - papers_sources/Research-Paper-Notes/PanGu-π.md
  - notes/PanGu-pi Nonlinearity Compensation.md
created: 2026-04-08
updated: 2026-04-08
summary: 级联激活补偿通过串联激活与可学习仿射变换增强 FFN 非线性表达，在有限开销下提升性能。
provenance:
  extracted: 0.85
  inferred: 0.13
  ambiguous: 0.02
---

# Cascaded Activation Nonlinearity Compensation

## 定义

在 FFN 中引入级联激活函数族与可学习仿射参数，补足标准激活的表达瓶颈。

## 设计目标

- 提升非线性容量。
- 尽量保持训练稳定性与推理吞吐。^[inferred]

## 关联页面

- [[references/PanGu-pi Nonlinearity Compensation]]
- [[concepts/LLM Nonlinearity Compensation]]
- [[synthesis/Long-Context Architecture Without Full Attention]]

