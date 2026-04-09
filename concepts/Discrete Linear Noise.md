---
title: Discrete Linear Noise
category: concept
tags:
  - concept
  - adversarial-attack
  - query-efficient
  - semantic-segmentation
  - perturbation
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-08
updated: 2026-04-09
summary: 通过离散化并结构化为线性噪声（横向/纵向），DLA 压缩搜索空间并提升有限查询下的攻击效率。
provenance:
  extracted: 0.85
  inferred: 0.13
  ambiguous: 0.02
---

# Discrete Linear Noise

## 定义

将连续扰动空间压缩到离散符号空间，并采用线状结构（horizontal / vertical）生成候选扰动。

## 作用

- 降低参数空间复杂度。
- 比全图随机噪声更容易形成可复用更新方向。^[inferred]
- 相比块状 patch，视觉结构更细且更稳定。^[inferred]

## 在 DLA 中的位置

- 先在探索阶段提供强初始化。
- 再在校准阶段配合符号翻转做分层细化。

## 注意

“更不易感知”主要基于论文可视化比较，跨数据集和跨观察者的主观一致性仍待系统验证。^[ambiguous]

## 联网补充

- DLA 将扰动搜索压到 l∞ 球极值点附近，并用水平/垂直线状离散噪声做 exploration 与 calibration，以显著压缩搜索空间。
- 线状扰动的重要性不只在查询效率，还在于它比块状 patch 更不易察觉，同时仍能借助分割模型的上下文传播扩散攻击效果。

## 关联

- [[concepts/Perturbation Interaction]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]



