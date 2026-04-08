---
title: Implicit Long Convolution Parameterization
category: concept
tags:
  - concept
  - long-context
  - convolution
  - architecture
sources:
  - papers_sources/Hyena Hierarchy 2302.10866/3_hyena.tex
  - notes/Hyena Hierarchy Towards Larger Convolutional Language Models.md
created: 2026-04-08
updated: 2026-04-08
summary: 隐式长卷积参数化通过神经函数生成超长滤波器，解耦滤波器长度与参数规模，支撑高效长上下文建模。
provenance:
  extracted: 0.88
  inferred: 0.1
  ambiguous: 0.02
---

# Implicit Long Convolution Parameterization

## 定义

使用位置编码与小网络生成卷积核，而非显式存储整段长滤波器，从而降低参数开销。

## 实践关注点

- 需要稳定的窗口函数或谱域约束以避免训练不稳。^[inferred]
- 适合与 FFT 加速实现协同部署。

## 关联页面

- [[references/Hyena Hierarchy Towards Larger Convolutional Language Models]]
- [[concepts/Hyena Operator]]
- [[synthesis/Long-Context Architecture Without Full Attention]]

