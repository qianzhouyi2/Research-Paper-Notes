---
title: Hyena Hierarchy Towards Larger Convolutional Language Models
category: reference
tags:
  - paper
  - long-context
  - convolution
  - sequence-modeling
sources:
  - papers_sources/Hyena Hierarchy 2302.10866/main.tex
  - papers_sources/2302.10866.tar
created: 2026-04-08
updated: 2026-04-08
summary: Hyena 通过“隐式长卷积+数据控制门控”提供次二次注意力替代，在长上下文下兼顾建模质量与显著速度优势。
provenance:
  extracted: 0.9
  inferred: 0.09
  ambiguous: 0.01
---

# Hyena Hierarchy Towards Larger Convolutional Language Models

## 基本信息

- 年份：2023（ICML）
- 任务：长序列语言建模中的注意力替代
- 论文笔记：[[notes/Hyena Hierarchy Towards Larger Convolutional Language Models]]

## 核心主张

- 注意力的二次复杂度限制上下文长度扩展。
- Hyena 使用“长卷积 + 数据控制门控”的交织结构，提供次二次复杂度替代。
- 论文报告在 2K 序列训练算力下降约 20%，8K 时约 2x 速度优势，64K 时约 100x。 

## 方法摘要

- 通过 Hyena recurrence 递归混合卷积与门控，提升表达能力。
- 用隐式参数化长滤波器，解耦“滤波器长度”和“参数规模”。
- 以 drop-in 方式替代注意力层，保持因果建模能力。

## 细化方法锚点

- 数据控制门控：在全局混合中引入输入相关调制，弥补纯卷积选择性不足。
- 隐式长卷积：以函数生成超长滤波器，降低参数规模与内存压力。
- 长上下文工程：速度优势依赖内核实现质量和序列长度区间。

## 局限与注意

- 高效实现对内核工程质量依赖较高，跨框架复现速度收益可能波动。^[inferred]

## 关联页面

- [[concepts/Hyena Operator]]
- [[concepts/Data-Controlled Gating in Sequence Models]]
- [[concepts/Implicit Long Convolution Parameterization]]
- [[entities/Yoshua Bengio]]
- [[entities/Michael Poli]]
- [[entities/Tri Dao]]
- [[synthesis/Long-Context Architecture Without Full Attention]]
