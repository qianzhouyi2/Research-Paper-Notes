---
title: Continual Backpropagation
category: concept
tags:
  - concept
  - continual-learning
  - optimization
  - plasticity
sources:
  - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv.tex
  - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv_2306.13812_source.tar.gz
created: 2026-04-08
updated: 2026-04-09
summary: Continual Backpropagation 在常规反向传播基础上持续替换低效隐藏单元，以低额外开销维持模型长期可塑性。
provenance:
  extracted: 0.9
  inferred: 0.09
  ambiguous: 0.01
---

# Continual Backpropagation

## 定义

CBP 是在标准反向传播之外，持续执行“低效单元选择性重初始化”的训练机制。

## 关键机制

- 维护隐藏单元效用统计。
- 只在成熟阈值之后替换低效单元，避免频繁抖动。
- 新单元输出连接置零，降低替换瞬间性能冲击。

## 适用场景

- 长时程持续学习任务。
- 关注“新任务学习能力”而非仅关注旧任务保持。

## 联网补充

- Nature 2024 论文把 continual backpropagation 定义为对标准反传的轻量改造：持续重初始化一小部分 less-used units，以长期维持 plasticity。
- 关键点不只是“重置单元”，而是持续向网络注入多样性；作者明确指出仅靠 gradient descent 本身不足以长期维持可塑性。

## 关联页面

- [[concepts/Loss of Plasticity in Continual Learning]]
- [[concepts/Feature Effective Rank Diagnostics]]
- [[concepts/Selective Neuron Reinitialization]]
- [[references/Maintaining Plasticity in Deep Continual Learning]]
- [[notes/Maintaining Plasticity in Deep Continual Learning]]
- [[synthesis/Continual Learning Plasticity Maintenance Playbook]]


