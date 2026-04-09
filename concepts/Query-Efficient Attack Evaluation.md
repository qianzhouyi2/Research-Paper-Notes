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
  - https://proceedings.mlr.press/v80/ilyas18a.html
  - https://arxiv.org/abs/1905.07121
  - https://arxiv.org/abs/1912.00049
created: 2026-04-08
updated: 2026-04-09
summary: 用固定查询预算比较攻击后 mIoU，可评估分割模型在现实黑盒条件下的鲁棒性与被攻破速度，并区分反馈接口与搜索空间设计的影响。
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

## 联网补充

- DLA 在 Cityscapes 上 50 次查询就能把 PSPNet 的 mIoU 从 77.83% 压到 2.14%，说明 query efficiency 很大程度来自搜索空间设计，而不只是梯度估计技巧。
- 但查询数不能脱离扰动形状单独解读；在 segmentation 里，攻击强度、可感知性和 query budget 是联动指标。
- ICML 2018 的 NES 黑盒攻击工作说明：即使能估计梯度，query efficiency 仍会受维度和采样方差限制，因此“更好的搜索空间”与“更好的梯度估计”是两条不同优化路径。
- 2019-05（arXiv:1905.07121）SimBA 展示了“随机方向双向试探”可在不显式估计完整梯度的情况下实现高查询效率。
- 2019-12（arXiv:1912.00049）Square Attack 进一步表明“局部方块随机搜索”在 score-based black-box 设定下可兼顾效率与约束控制。

## 关联

- [[concepts/Natural Evolutionary Strategies (NES)]]
- [[concepts/Proxy Index mIoU Optimization]]
- [[concepts/SimBA (Simple Black-box Attack)]]
- [[concepts/Square Attack]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[entities/FCN]]
- [[entities/PSPNet]]
- [[entities/DeepLabv3]]
- [[entities/SegFormer]]
- [[entities/MaskFormer]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]



