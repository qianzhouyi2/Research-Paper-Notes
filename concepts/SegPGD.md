---
title: SegPGD
category: concept
tags:
  - concept
  - adversarial-attack
  - semantic-segmentation
  - robustness
  - optimization
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh.md
  - https://arxiv.org/abs/2207.12391
  - https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890306.pdf
created: 2026-04-09
updated: 2026-04-10
summary: SegPGD 是面向语义分割的高效 PGD 变体，通过动态重加权正确与错误像素损失，在相同步数下生成更强攻击并支撑更强对抗训练。
provenance:
  extracted: 0.81
  inferred: 0.17
  ambiguous: 0.02
---

# SegPGD

## 定义

SegPGD 是专门针对语义分割设计的 PGD 变体，目标是在较少攻击迭代下更快地误导更多像素预测。

## 核心机制

- 论文把像素分成“当前仍预测正确”和“已经被攻破”的两组。
- 标准 PGD 中，已被攻破像素往往带来较大的损失，容易主导梯度，但继续放大这部分损失不一定能进一步提升攻击效果。
- SegPGD 对这两组像素的损失做动态加权：前期更关注还没攻破的像素，后期再逐渐提高对已错像素的权重，以避免它们重新变回正确预测。
- 论文最终采用简单的线性权重调度。 

## 为什么比普通 PGD 更适合分割

- 分割是密集预测任务，攻击目标不是单个标签，而是尽可能多像素同时出错。
- 如果所有像素损失一视同仁，梯度会被“已经错的像素”拖偏，导致需要更多步数才能继续攻破剩余正确像素。
- SegPGD 通过重加权减少这种梯度浪费，因此在相同步数下往往比 PGD 更强。

## 作用

- 作为白盒攻击：更适合评估分割模型是否真的鲁棒。
- 作为对抗训练底层攻击：SegPGD-AT 往往比 PGD-AT 产出更强的鲁棒基线。

## 联网补充

- ECCV 2022 论文把 SegPGD 定义为“effective and efficient segmentation attack”，并给出收敛分析说明其在同样迭代步数下可比 PGD 生成更强样本。
- 论文实验显示，动态线性权重调度比固定权重或简单只看正确像素的方案更稳定。

## 关联

- [[concepts/Semantic Segmentation]]
- [[concepts/Indirect Local Attack in Segmentation]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Proxy Index mIoU Optimization]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: SegPGD 是专门针对语义分割设计的 PGD 变体，目标是在较少攻击迭代下更快地误导更多像素预测。
- Primary online sources used in this pass:
- https://arxiv.org/abs/2207.12391
- https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890306.pdf
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

