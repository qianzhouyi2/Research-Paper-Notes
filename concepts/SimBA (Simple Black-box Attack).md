---
title: SimBA (Simple Black-box Attack)
category: concept
tags:
  - concept
  - adversarial-attack
  - black-box-attack
  - query-efficient
  - robustness
sources:
  - https://arxiv.org/abs/1905.07121
created: 2026-04-09
updated: 2026-04-10
summary: SimBA 通过逐方向随机试探并保留有利更新，在无需梯度的黑盒设置下实现高查询效率攻击。
provenance:
  extracted: 0.82
  inferred: 0.15
  ambiguous: 0.03
---

# SimBA (Simple Black-box Attack)

## 定义

SimBA 是 score-based 黑盒对抗攻击方法。它不估计完整梯度，而是沿随机基方向做“正负两次试探”，保留让攻击目标更优的更新。

## 基本流程

1. 选取方向 \(q\)（像素基或 DCT 基）。
2. 查询 \(x+\epsilon q\) 与 \(x-\epsilon q\)。
3. 保留更有利的一侧（例如非定向攻击里降低真类分数）。
4. 迭代直到成功或预算耗尽。

## 方法特点

- 实现简单，不需要显式梯度估计。
- 在有限查询预算下通常优于大量早期随机攻击基线。^[inferred]
- SimBA-DCT 常比纯像素方向更省查询。^[inferred]

## 联网补充

- 2019-05（arXiv:1905.07121）：论文直接将方法命名为 “Simple Black-box Adversarial Attacks”，核心贡献就是“简化攻击步骤 + 提升查询效率”。
- 该方法属于 score-based 路线，适配能返回置信度/分数的 API；若仅有 top-1 标签则通常需转为 decision-based 方法。^[inferred]

## 关联

- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/Square Attack]]
- [[concepts/SignSGD]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
