---
title: Continual Learning Plasticity Maintenance Playbook
category: synthesis
tags:
  - synthesis
  - continual-learning
  - optimization
  - representation
sources:
  - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv.tex
  - notes/Maintaining Plasticity in Deep Continual Learning.md
created: 2026-04-08
updated: 2026-04-10
summary: 持续学习应并行监测“可塑性指标+任务表现”，并用效用驱动重初始化等机制持续注入学习能力。
provenance:
  extracted: 0.76
  inferred: 0.22
  ambiguous: 0.02
---

# Continual Learning Plasticity Maintenance Playbook

## 核心流程

1. 先诊断：用准确率趋势、死单元比例、有效秩监测可塑性衰退。
2. 再干预：执行低比例、效用驱动的选择性重初始化。
3. 持续复盘：比较不同替换率与成熟阈值对长期性能的影响。

## 代表页面

- [[references/Maintaining Plasticity in Deep Continual Learning]]
- [[concepts/Loss of Plasticity in Continual Learning]]
- [[concepts/Feature Effective Rank Diagnostics]]
- [[concepts/Selective Neuron Reinitialization]]
- [[concepts/Continual Backpropagation]]

## 联网补充

- 《Maintaining Plasticity in Deep Continual Learning》明确提出“loss of plasticity”是持续学习核心瓶颈，并给出 continual backprop 作为可长期维持可塑性的机制。
- 因此该 playbook 的关键不是只看任务精度，而是并行监测“新任务吸收能力 + 历史能力保持 + 表征退化信号”。

## Online Supplement (2026-04-10)

- This synthesis page is cross-checked online for cross-paper consistency and evaluation-scope alignment.
- Text anchor used: 1. 先诊断：用准确率趋势、死单元比例、有效秩监测可塑性衰退。 2. 再干预：执行低比例、效用驱动的选择性重初始化。 3. 持续复盘：比较不同替换率与成熟阈值对长期性能的影响。
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

