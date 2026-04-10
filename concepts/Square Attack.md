---
title: Square Attack
category: concept
tags:
  - concept
  - adversarial-attack
  - black-box-attack
  - query-efficient
  - optimization
sources:
  - https://arxiv.org/abs/1912.00049
created: 2026-04-09
updated: 2026-04-10
summary: Square Attack 通过局部方块随机搜索更新，在 L2/Linf 约束下实现高效的分数型黑盒攻击。
provenance:
  extracted: 0.83
  inferred: 0.14
  ambiguous: 0.03
---

# Square Attack

## 定义

Square Attack 是 score-based 黑盒攻击：每次仅在一个局部方块内注入随机扰动，并按目标函数是否改进来接受或拒绝更新。

## 基本流程

1. 在当前样本附近采样一个方块位置与大小。
2. 只改动该方块像素（随机符号/幅度）。
3. 查询模型分数并判断是否接受更新。
4. 重复直到攻击成功或查询预算耗尽。

## 方法特点

- 不依赖梯度与替代模型，直接做随机搜索。
- 同时支持 \(L_2\) 与 \(L_\infty\) 约束设置。
- 单次更新覆盖局部结构，通常比逐像素试探更高效。^[inferred]

## 联网补充

- 2019-12（arXiv:1912.00049）：Square Attack 明确提出“query-efficient black-box adversarial attack via random search”。
- 该路线与 SimBA 的“方向试探”不同，更强调“局部结构化扰动 + 接受拒绝搜索”。

## 关联

- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/SimBA (Simple Black-box Attack)]]
- [[concepts/SignSGD]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
