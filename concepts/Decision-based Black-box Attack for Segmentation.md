---
title: Decision-based Black-box Attack for Segmentation
category: concept
tags:
  - concept
  - adversarial-attack
  - black-box-attack
  - semantic-segmentation
  - robustness
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
  - https://arxiv.org/abs/2402.01220
created: 2026-04-08
updated: 2026-04-09
summary: 语义分割中的 decision-based 攻击仅依赖标签输出，核心挑战来自像素级多约束优化和有限查询预算。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Decision-based Black-box Attack for Segmentation

## 定义

攻击者只能访问分割模型输出标签（无概率、无梯度），目标是在查询预算内最大化像素误分并降低整体指标（常用 mIoU）。

## 与图像分类差异

- 图像分类通常是单标签决策。
- 语义分割是像素级决策，等价于大规模多约束问题。
- “先跨边界再缩扰动”在分割任务中不稳定，可能反向提升 mIoU。

## 典型困难

- 优化目标不一致。
- 扰动交互导致局部反复。
- 参数空间巨大且查询昂贵。

## 设计启发

- 先定义稳定代理指标，再做可控搜索空间压缩。
- 局部结构化扰动比全图随机噪声更有效。^[inferred]
- score-based 基线常走 [[concepts/Natural Evolutionary Strategies (NES)|NES]] 这类梯度估计路线；decision-based 分割攻击更依赖可比较代理指标和 accept/reject 搜索。^[inferred]

## 联网补充

- arXiv 2402.01220 将其定义为语义分割上的首个 decision-based black-box 系统研究，并把 DLA 建立在 random search、proxy index 与 discrete linear noise 之上。
- 论文强调该问题不同于图像分类黑盒攻击：像素级输出把它变成大规模多约束优化，许多分类攻击套路不能直接照搬。
- 这也解释了为什么 DLA 会同时讨论 NES、Bandits、SimBA、Square 等 score-based baseline：它们提供了 query efficiency 参照，但并不等价于“只返回标签”的真正 decision-based setting。^[inferred]
- 在进入 query-based 黑盒设置前，语义分割攻击主线已经给出两个重要先验：[[concepts/Indirect Local Attack in Segmentation]] 说明局部扰动可借长程上下文影响远处像素；[[concepts/SegPGD]] 说明分割攻击要显式处理“正确/错误像素损失失衡”问题。^[inferred]

## 关联

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Natural Evolutionary Strategies (NES)]]
- [[concepts/L-infinity Norm Ball]]
- [[concepts/Indirect Local Attack in Segmentation]]
- [[concepts/SegPGD]]
- [[concepts/Proxy Index mIoU Optimization]]
- [[concepts/Perturbation Interaction]]
- [[concepts/Discrete Linear Noise]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]



