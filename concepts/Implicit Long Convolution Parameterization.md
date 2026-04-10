---
title: "Implicit Long Convolution Parameterization"
category: concept
tags:
  - concept
sources:
  - workspace/wiki-update-2026-04-10-global-lint-remediation
created: 2026-04-10
updated: 2026-04-10
summary: "﻿---"
---
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
updated: 2026-04-10
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

## 联网补充

- Hyena 用隐式参数化生成长卷积核，核心收益是把“感受野长度”与“直接参数量”解耦，从而支持数千到数十万 token 的全局混合。
- 但隐式长卷积主要解决的是可扩展性；若没有门控或其他输入条件机制，它本身并不自动具备足够强的内容选择性。

## 关联页面

- [[references/Hyena Hierarchy Towards Larger Convolutional Language Models]]
- [[concepts/Hyena Operator]]
- [[synthesis/Long-Context Architecture Without Full Attention]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Implicit Long Convolution Parameterization category: concept tags: - concept - long-context - convolution - architecture sources: - papers_sources/Hyena Hierarchy 2302.10866/3_hyena.tex - notes/Hyena Hierarchy Towards Larger Convolutional Language ...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

