---
title: L-infinity Norm Ball
category: concept
tags:
  - concept
  - geometry
  - optimization
  - adversarial-attack
  - robustness
sources:
  - https://proceedings.mlr.press/v97/moon19a.html
  - https://arxiv.org/abs/1811.10828
  - https://openreview.net/forum?id=rJzIBfZAb
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-09
updated: 2026-04-10
summary: L-infinity 范数球约束每个坐标的最大改变量；在对抗攻击里，其顶点对应每一维都取到 ±epsilon 的符号向量。
provenance:
  extracted: 0.83
  inferred: 0.15
  ambiguous: 0.02
---

# L-infinity Norm Ball

## 定义

以输入 `x` 为中心、半径为 `\epsilon` 的 `L_\infty` 范数球写作：

\[
B_\infty(x, \epsilon) = \{x' \mid \|x' - x\|_\infty \le \epsilon\}
\]

其中 `\|v\|_\infty = max_i |v_i|`，也就是所有分量里绝对值最大的那个。

## 直观理解

- 在 2 维里，它是正方形而不是圆。
- 在 3 维里，它是立方体而不是球。
- 在高维里，它是超立方体。^[inferred]
- 它约束的是“每个像素最多能改多少”，不是“所有像素总共改了多少”。

## 顶点 / 极值点

- `L_\infty` 球的顶点对应每一维都取到边界，因此可写成 `x + \delta`，其中 `\delta_i \in \{-\epsilon, +\epsilon\}`。
- 这就是很多离散符号攻击里常见的 `\{-\epsilon, +\epsilon\}^d`。

## 在对抗攻击里的意义

- 它是最常见的 threat model 之一，因为它直接限制单像素最大扰动。^[inferred]
- 许多 `L_\infty` 攻击都会把解推到约束边界；而把搜索进一步压到顶点，是一种更激进的离散化近似。

## 联网补充

- Moon et al. (ICML 2019) 对 `L_\infty` 顶点化给出了更直接的动机：在线性 surrogate 下，最优解落在 `L_\infty` 球的 vertex；论文同时报告 PGD 得到的样本也常位于这些顶点。
- Chen et al. (AAAI 2020, Frank-Wolfe) 的表述更偏“边界”而不是“所有解都在顶点”，因此它支持的是 boundary-biased intuition，而不是更强的 vertex 定理。
- DLA 利用这一经验，把连续空间 `[-\epsilon, \epsilon]^d` 直接压成 `\{-\epsilon, \epsilon\}^d`，再进一步限制为线状离散噪声。

## 关联

- [[concepts/Discrete Linear Noise]]
- [[concepts/Natural Evolutionary Strategies (NES)]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: 以输入 `x` 为中心、半径为 `\epsilon` 的 `L_\infty` 范数球写作：
- Primary online sources used in this pass:
- https://proceedings.mlr.press/v97/moon19a.html
- https://arxiv.org/abs/1811.10828
- https://openreview.net/forum?id=rJzIBfZAb
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

