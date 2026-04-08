---
title: MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification
category: reference
tags:
  - paper
  - adversarial-robustness
  - post-training
  - feature-sparsification
sources:
  - papers_sources/MeanSparse 2406.05927/main.tex
  - papers_sources/MeanSparse 2406.05927/source.tar.gz
created: 2026-04-08
updated: 2026-04-08
summary: MeanSparse 在不重训模型的前提下，通过均值中心化稀疏化抑制低信息特征波动，并在多个基准上提升 AutoAttack 鲁棒准确率。
provenance:
  extracted: 0.9
  inferred: 0.09
  ambiguous: 0.01
---

# MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification

## 基本信息

- 年份：2024（arXiv）
- 任务：对抗训练后模型的后处理鲁棒增强
- 论文笔记：[[notes/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature]]

## 核心主张

- 对抗训练后仍存在可被攻击利用的低信息特征波动。
- 在激活前加入均值中心化稀疏化算子，可在几乎零重训成本下提升鲁棒性。
- 论文报告 RobustBench 头部模型的 AutoAttack 指标有稳定提升（如 CIFAR-10: 73.71% -> 75.28%）。

## 方法摘要

- 先统计每个通道的均值和标准差。
- 设阈值 `T_th = alpha * sigma`，将均值邻域内特征回填到均值。
- 采用 post-training 集成而非训练内集成，避免梯度传播不稳定。

## 细化方法锚点

- 理论动机：从稀疏正则化近端算子推导到硬阈值回填。
- 通道自适应：逐通道统计比全局统计更稳健。
- 部署形态：无需重训主干，适配现有鲁棒模型作为后处理模块。

## 局限与注意

- 阈值搜索仍需针对模型和任务调节。^[inferred]

## 关联页面

- [[concepts/Mean-Centered Feature Sparsification]]
- [[synthesis/Adversarial Robustness Evaluation Patterns]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]
- [[entities/Amir Houmansadr]]
- [[entities/Mohammadreza Teymoorianfard]]
- [[entities/Shiqing Ma]]
- [[references/Are aligned neural networks adversarially aligned]]
