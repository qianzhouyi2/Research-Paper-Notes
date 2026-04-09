---
title: Natural Evolutionary Strategies (NES)
category: concept
tags:
  - concept
  - optimization
  - adversarial-attack
  - black-box-attack
  - query-efficient
sources:
  - https://jmlr.org/papers/v15/wierstra14a.html
  - https://proceedings.mlr.press/v80/ilyas18a.html
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-09
updated: 2026-04-09
summary: NES 通过优化搜索分布并用采样估计自然梯度，在黑盒攻击中常被用作高维无梯度梯度估计器。
provenance:
  extracted: 0.8
  inferred: 0.17
  ambiguous: 0.03
---

# Natural Evolutionary Strategies (NES)

## 定义

NES 不是直接优化单个点，而是优化一个搜索分布 `p_\theta(z)`，目标通常写成 `J(\theta) = E_{z ~ p_\theta}[f(z)]`。更新时使用 natural gradient，而不是普通参数梯度。

## 在黑盒攻击里的常见用法

- 在输入 `x` 附近采样高斯方向 `u_i`。
- 比较 `f(x + \sigma u_i)` 与 `f(x - \sigma u_i)` 的变化。
- 用这些采样结果加权平均，估计一个梯度方向。
- 再把这个估计方向用于后续攻击更新。^[inferred]

## 直观理解

- 拿不到真实梯度时，就沿很多随机方向各试一次。
- 哪些方向让目标函数更优，就说明真实梯度更可能朝那边。
- NES 把“方向试探”组织成了一个分布优化问题。^[inferred]

## 在黑盒攻击里的边界

- 更适合 [[concepts/Query-Efficient Attack Evaluation|score-based 黑盒攻击]]，因为通常需要可比较的分数或损失。
- 若接口只返回最终标签，NES 就不如 decision-based 的 accept/reject 搜索直接。^[inferred]
- 维度越高、采样方差越大，查询成本通常越高。

## 联网补充

- JMLR 2014 的原始 NES 工作把它表述为“在搜索分布上最大化期望 fitness，并用 natural gradient 更新分布参数”。
- ICML 2018 的黑盒攻击论文把 NES 明确用作 query-limited attack 的梯度估计器，并采用 antithetic sampling 降低估计方差。
- 在 DLA 这篇分割攻击论文里，NES 的角色主要是 score-based baseline；DLA 的改进点不在更精细的梯度估计，而在于把搜索空间离散化并结构化。^[inferred]

## 关联

- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/L-infinity Norm Ball]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
