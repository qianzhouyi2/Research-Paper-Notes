---
title: Query-Efficient Attack Evaluation
category: concept
tags:
  - concept
  - evaluation
  - query-efficient
  - robustness
  - semantic-segmentation
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-08
updated: 2026-04-08
summary: 用固定查询预算比较攻击后 mIoU，可评估分割模型在现实黑盒条件下的鲁棒性与被攻破速度。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Query-Efficient Attack Evaluation

## 目的

在现实攻击场景中，查询次数通常受限。评估应报告“在 N 次查询内能把性能打到多低”。

## 典型预算

- 小预算：10 queries（高隐蔽、高约束）
- 中预算：50 queries
- 较大预算：200 queries

## 指标实践

- 主指标：攻击后 mIoU（越低越易被攻破）
- 可补充：不同 epsilon 下的退化曲线

## 结论模式

若方法在低查询预算下仍能显著降 mIoU，则更接近工业威胁模型。^[inferred]

## 关联

- [[concepts/Proxy Index mIoU Optimization]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[entities/SegFormer]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]

