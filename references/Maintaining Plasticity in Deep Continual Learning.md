---
title: Maintaining Plasticity in Deep Continual Learning
category: reference
tags:
  - paper
  - continual-learning
  - optimization
  - representation
sources:
  - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv.tex
  - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv_2306.13812_source.tar.gz
created: 2026-04-08
updated: 2026-04-08
summary: 论文系统刻画深度持续学习中的“可塑性丧失”，并提出 Continual Backpropagation 以选择性重置低效单元，显著缓解长期退化。
provenance:
  extracted: 0.89
  inferred: 0.1
  ambiguous: 0.01
---

# Maintaining Plasticity in Deep Continual Learning

## 基本信息

- 年份：2023（arXiv，后续有 Nature 版本）
- 任务：持续学习中的 loss of plasticity
- 论文笔记：[[notes/Maintaining Plasticity in Deep Continual Learning]]
- 联网核验：Nature 正式版以 “Loss of plasticity in deep continual learning” 发表，作者在预印本基础上补入 Qingfeng Lan。

## 核心主张

- 深度网络在长期持续学习中不仅会遗忘，还会逐渐“学不动新任务”。
- 在 continual ImageNet 设定中，早期任务准确率约 89%，到第 2000 任务下降到约 77%。
- Continual Backpropagation（CBP）通过持续重置低效单元来维持可塑性。

## 方法摘要

- 每步常规反向传播后，维护单元效用统计。
- 选择低效用且达到成熟阈值的隐藏单元重初始化。
- 新单元输出权重置零，以减少即时功能扰动。

## 细化方法锚点

- 诊断层：用有效秩、死单元比例等指标区分“遗忘”与“可塑性衰退”。
- 干预层：效用驱动替换优于随机替换，且成熟阈值可减小抖动。
- 预算层：以低替换率持续注入可学习能力，降低长期退化。^[inferred]

## 局限与注意

- 效用定义和替换策略具有一定启发式性质，在更大模型上的最优性仍需验证。^[ambiguous]

## 关联页面

- [[concepts/Loss of Plasticity in Continual Learning]]
- [[concepts/Continual Backpropagation]]
- [[concepts/Feature Effective Rank Diagnostics]]
- [[concepts/Selective Neuron Reinitialization]]
- [[entities/Shibhansh Dohare]]
- [[entities/Richard S. Sutton]]
- [[synthesis/Continual Learning Plasticity Maintenance Playbook]]
