---
title: "Selective Neuron Reinitialization"
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
title: Selective Neuron Reinitialization
category: concept
tags:
  - concept
  - continual-learning
  - optimization
  - plasticity
sources:
  - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv.tex
  - notes/Maintaining Plasticity in Deep Continual Learning.md
created: 2026-04-08
updated: 2026-04-10
summary: 选择性神经元重初始化通过替换低效隐藏单元并保护新单元成熟期，持续注入学习能力以减缓可塑性衰退。
provenance:
  extracted: 0.89
  inferred: 0.09
  ambiguous: 0.02
---

# Selective Neuron Reinitialization

## 定义

按效用分数和成熟阈值选择隐藏单元执行重置，输入权重重采样、输出权重置零，降低瞬时扰动。

## 关键超参数

- 替换率 `rho`
- 成熟阈值 `m`
- 效用统计平滑系数 `eta`

## 联网补充

- Nature 论文强调 continual backpropagation 每次只重初始化一小部分 less-used units，因此“selective”本身就是稳定性来源，而不是实现细节。
- 成熟阈值和输出权重置零的设计同样关键：没有这些保护，新单元注入会更像噪声重置而不是可控的多样性补给。

## 关联页面

- [[references/Maintaining Plasticity in Deep Continual Learning]]
- [[concepts/Continual Backpropagation]]
- [[synthesis/Continual Learning Plasticity Maintenance Playbook]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Selective Neuron Reinitialization category: concept tags: - concept - continual-learning - optimization - plasticity sources: - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv.tex - notes/Maintaining Plasticity in Deep Contin...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

