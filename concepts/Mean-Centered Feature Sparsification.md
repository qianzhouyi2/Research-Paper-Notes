---
title: Mean-Centered Feature Sparsification
category: concept
tags:
  - concept
  - adversarial-robustness
  - feature-engineering
  - post-training
sources:
  - papers_sources/MeanSparse 2406.05927/main.tex
  - papers_sources/MeanSparse 2406.05927/source.tar.gz
created: 2026-04-08
updated: 2026-04-08
summary: 均值中心化稀疏化通过在通道均值邻域内压平低信息波动，减少对抗扰动可利用空间，是一种可插拔后处理鲁棒增强策略。
provenance:
  extracted: 0.9
  inferred: 0.08
  ambiguous: 0.02
---

# Mean-Centered Feature Sparsification

## 定义

对激活前特征按通道统计均值与标准差，并在 `mu +/- alpha*sigma` 邻域内将特征回填到均值。

## 作用机制

- 抑制均值附近的低信息小幅波动。
- 降低攻击者通过细粒度扰动操控特征的空间。
- 通过后处理插入，不改原模型参数。

## 工程要点

- 阈值通常写为 `T_th = alpha * sigma`。
- 采用逐通道统计优于全局统一统计。
- 需要在 clean 与 robust 指标之间搜索平衡阈值。^[inferred]

## 关联页面

- [[references/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification]]
- [[synthesis/Adversarial Robustness Evaluation Patterns]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]
- [[entities/Amir Houmansadr]]
- [[notes/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature]]
