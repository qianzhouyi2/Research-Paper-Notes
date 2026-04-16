---
title: "Robust Training Strategies for Semantic Segmentation"
category: synthesis
tags:
  - synthesis
  - semantic-segmentation
  - adversarial-robustness
  - efficiency
sources:
  - workspace/wiki-update-2026-04-10-cn-localization
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - https://arxiv.org/abs/2010.05495
  - https://arxiv.org/abs/2003.06555
  - https://arxiv.org/abs/2207.12391
  - https://arxiv.org/abs/2306.12941
created: 2026-04-10
updated: 2026-04-16
summary: "跨论文总结：鲁棒语义分割训练不只是加大 attack budget，更关键的是把 initialization 当成一等变量，并在强评测协议下比较真实收益。"
provenance:
  extracted: 0.79
  inferred: 0.18
  ambiguous: 0.03
---

# Robust Training Strategies for Semantic Segmentation

## 核心结论

- 语义分割中的 adversarial training 比分类更难，因为攻击目标更复杂、训练成本更高，而且弱评测容易给出虚假安全感。
- 训练策略不能只比较“用了多少步 PGD”，还要比较 initialization、评测协议和训练轮数。
- 初始化在 segmentation adversarial training 里是一个一等变量，而不只是训练前的小细节。
- [[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models|Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]] 把 robust initialization 提升为一等变量，是这条线上非常关键的推进。

## 本轮补充（2026-04-16）

- PIR-AT 的核心启示是：在 segmentation 里，好的鲁棒起点往往比更长训练更重要。
- 论文显示 `2-step PIR-AT, 50 epochs` 能超过 `2-step AT, 300 epochs`；在 Ade20K 上 `32 epochs PIR-AT` 也能超过 `128 epochs AT`。
- 训练时作者仍使用标准 CE + PGD，而不是直接把更强攻击 loss 搬进训练，这说明“更强评测 loss”不一定等于“更好训练 loss”。
- PIR-AT 的收益还依赖可用且兼容的 robust pretrained backbone，因此它更像一条 initialization-first recipe，而不是可以无条件迁移到任何架构的通用定律。

## 覆盖论文

- [[references/Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers]]
- [[references/Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation]]
- [[references/SegPGD - An Effective and Efficient Adversarial Attack for Segmentation|SegPGD: An Effective and Efficient Adversarial Attack for Segmentation]]
- [[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]]
- [[references/RP-PGD - Enhancing Semantic Segmentation Robustness via Region-based Prioritized PGD|RP-PGD: Enhancing Semantic Segmentation Robustness via Region-based Prioritized PGD]]
- [[references/Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness (arXiv preprint)|Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness (arXiv preprint)]]

## 可复用模式

- 将 initialization 作为显式对照变量单独做 ablation。
- 把训练 cost 以 epoch 数或 wall-clock 成本纳入主结果，而不是只汇报 robustness。
- 不要默认“更强攻击 loss”就应该直接成为训练 loss，先验证它是否真的改善训练动态。^[inferred]
- 训练论文的最终结论必须在强评测协议下复核。^[inferred]

## 关联概念与链接

- [[concepts/Prior-informed Robust Adversarial Training (PIR-AT)]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]
- [[concepts/Divide-and-Conquer Adversarial Training for Segmentation]]
- [[concepts/SegPGD]]
