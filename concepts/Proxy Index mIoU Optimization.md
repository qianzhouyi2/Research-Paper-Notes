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
updated: 2026-04-10
summary: 在 decision-based 分割攻击中，以 mIoU 作为代理指标可提供稳定反馈，优于仅用像素准确率的局部目标。
provenance:
  extracted: 0.62
  inferred: 0.38
  ambiguous: 0.00
---

# Proxy Index mIoU Optimization

## 概念

在无法访问梯度和置信度时，用可观测指标驱动搜索。DLA 选择 mIoU 作为代理指标，每次只接受能进一步降低 mIoU 的扰动更新。^[extracted]

## Random Attack 作为最小基线

- 从干净图像出发，每轮随机采样一个小扰动，并把结果投影回 `\epsilon` 邻域。
- 只要新的代理指标更小，就接受这次更新；否则丢弃。^[extracted]
- 因而它本质上是“随机搜索 + 贪心接受”的黑盒基线，而不是显式梯度估计。^[inferred]

## 为什么是 mIoU

- 比 PAcc 更能反映分割的整体结构质量。^[extracted]
- 与数据集评估口径一致，便于跨样本比较。^[extracted]

## 影响

- 将“更新是否有效”变成可比较的单调准则。
- 降低随机搜索中的无效查询比例。^[inferred]

## 风险

- 如果目标是特定类别破坏，单一 mIoU 可能过粗。
- 代理指标与真实任务目标不一致时，会产生优化偏差。^[inferred]

## 联网补充

- 在 DLA 的 case study 里，mIoU 被选为 proxy index，不是因为它最容易算，而是因为它比 PAcc 更贴近分割整体结构质量和 benchmark 评估口径。^[extracted]
- 因此代理指标的价值在于提供更有信息量的 accept/reject 信号；若 proxy 和真实目标脱节，再高频查询也只是低效搜索。

## 关联

- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/L-infinity Norm Ball]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[entities/SegFormer]]
