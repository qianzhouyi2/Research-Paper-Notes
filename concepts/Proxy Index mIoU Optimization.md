---
title: Proxy Index mIoU Optimization
category: concept
tags:
  - concept
  - optimization
  - semantic-segmentation
  - evaluation
  - adversarial-attack
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-08
updated: 2026-04-08
summary: 在 decision-based 分割攻击中，以 mIoU 作为代理指标可提供稳定反馈，优于仅用像素准确率的局部目标。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# Proxy Index mIoU Optimization

## 概念

在无法访问梯度和置信度时，用可观测指标驱动搜索。DLA 选择 mIoU 作为代理指标，每次只接受能进一步降低 mIoU 的扰动更新。

## 为什么是 mIoU

- 比 PAcc 更能反映分割的整体结构质量。
- 与数据集评估口径一致，便于跨样本比较。

## 影响

- 将“更新是否有效”变成可比较的单调准则。
- 降低随机搜索中的无效查询比例。^[inferred]

## 风险

- 如果目标是特定类别破坏，单一 mIoU 可能过粗。
- 代理指标与真实任务目标不一致时，会产生优化偏差。^[inferred]

## 关联

- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[entities/SegFormer]]

