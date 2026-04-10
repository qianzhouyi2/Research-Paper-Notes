---
title: Feature Effective Rank Diagnostics
category: concept
tags:
  - concept
  - continual-learning
  - representation
  - diagnostics
sources:
  - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv.tex
  - notes/Maintaining Plasticity in Deep Continual Learning.md
created: 2026-04-08
updated: 2026-04-10
summary: 有效秩诊断用表示谱熵近似衡量特征多样性退化，可用于监测持续学习中的可塑性丧失。
provenance:
  extracted: 0.9
  inferred: 0.08
  ambiguous: 0.02
---

# Feature Effective Rank Diagnostics

## 定义

用特征协方差谱分布的熵定义有效秩，刻画表示空间是否逐步坍塌。

## 用法

- 与任务准确率联合跟踪，区分“暂时波动”与“长期可塑性衰退”。
- 可作为持续学习算法调参的早期告警信号。^[inferred]

## 联网补充

- Nature 论文把 effective rank 的持续下降列为 loss of plasticity 的三个关键相关量之一，并指出它会限制新任务开始时可表示的解空间。
- 因此 effective rank 更适合作为容量健康度诊断，而不是单独的训练目标；它需要与 dead units 和 weight magnitude 一起解释。

## 关联页面

- [[references/Maintaining Plasticity in Deep Continual Learning]]
- [[concepts/Loss of Plasticity in Continual Learning]]
- [[synthesis/Continual Learning Plasticity Maintenance Playbook]]
