---
title: Perturbation Interaction
category: concept
tags:
  - concept
  - adversarial-attack
  - optimization
  - semantic-segmentation
  - perturbation
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-08
updated: 2026-04-08
summary: 扰动交互指迭代更新之间相互抵消，导致分割攻击出现“本轮成功、下轮回退”的优化不稳定现象。
provenance:
  extracted: 0.87
  inferred: 0.11
  ambiguous: 0.02
---

# Perturbation Interaction

## 定义

在语义分割攻击中，新加入的扰动可能抵消或破坏先前扰动的效果，导致像素类别在相邻迭代之间反复切换。

## 现象

- 同一像素在第 t 步被攻击成功，但在第 t+1 步恢复正确类别。
- 优化过程更易陷入局部最优。

## 应对策略

- 从全图随机更新转向局部结构化更新。
- 使用线性噪声并配合局部符号翻转。

## 工程含义

有限查询下，减少“互相打架”的更新比单次大幅扰动更关键。^[inferred]

## 关联

- [[concepts/Discrete Linear Noise]]
- [[concepts/Proxy Index mIoU Optimization]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]

