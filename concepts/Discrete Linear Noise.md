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
  - https://proceedings.mlr.press/v97/moon19a.html
  - https://arxiv.org/abs/1811.10828
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

- 离散部分对应 `\delta_i \in \{-\epsilon, +\epsilon\}`，也就是从 [[concepts/L-infinity Norm Ball|`L_\infty` 范数球]] 的顶点集合里搜索。
- 线性部分则进一步把自由度压到“按行”或“按列”的符号模式，而不是让每个像素独立变化。^[inferred]

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

- Moon et al. (ICML 2019) 对 “`L_\infty` 顶点化搜索” 给出了更直接的依据：在线性 surrogate 下，最优解出现在顶点，PGD 样本也常位于这些顶点。
- Chen et al. (AAAI 2020, Frank-Wolfe) 更偏向支持“对抗样本常位于约束边界”这一较弱结论；因此 DLA 这里更准确的动机是 boundary / vertex-biased search，而不是普适定理。
- 线状扰动的重要性不只在查询效率，还在于它比块状 patch 更不易察觉，同时仍能借助分割模型的上下文传播扩散攻击效果。

## 关联

- [[concepts/L-infinity Norm Ball]]
- [[concepts/Perturbation Interaction]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]


