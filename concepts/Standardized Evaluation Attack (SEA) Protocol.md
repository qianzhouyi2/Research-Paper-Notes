---
title: "Segmentation Ensemble Attack (SEA) Protocol"
aliases:
  - Standardized Evaluation Attack (SEA) Protocol
category: concept
tags:
  - concept
  - semantic-segmentation
  - adversarial-robustness
  - evaluation
  - methodology
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/25_Towards_Reliable_Evaluation_and_Fast_Training_of_Robust_Semantic_Segme/paper_resources/arxiv_2306.12941_translated.md
  - references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models.md
  - https://arxiv.org/abs/2306.12941
created: 2026-04-10
updated: 2026-04-16
summary: "SEA：面向语义分割鲁棒评测的白盒集成协议；它用三种互补损失配合 red-epsilon APGD 逼近更可靠的最坏情况，其中 red-epsilon 是优化技巧，不等于 SEA 本身。"
provenance:
  extracted: 0.8
  inferred: 0.17
  ambiguous: 0.03
---

# Segmentation Ensemble Attack (SEA) Protocol

## 定义

SEA 是语义分割鲁棒性评测协议，不是单一攻击，也不等于 `red-epsilon APGD` 这一条优化技巧。它把三种互补的 segmentation-specific attacks 集成为一个 worst-case evaluator：

- `L_MCE`：面向 average pixel accuracy
- `L_MCE-Bal`：面向 mIoU
- `L_JS`：通过 JS divergence 自动降权已错像素

每个 loss 各运行一次 `300` iterations 的 red-epsilon APGD，再对三者取最坏结果；其中 `red-epsilon APGD` 负责更稳地优化，SEA 负责把多路攻击合成为评测协议。

## 为什么需要它

- segmentation 攻击必须同时翻转大量像素，普通 CE / 单一 PGD 很容易被已经错分的像素拖偏。
- mIoU 和 accuracy 不是同一个攻击目标，单一 loss 容易只压一个指标。
- 论文证明 SegPGD 和 CosPGD 会高估若干模型的鲁棒性，而 SEA 能更稳定地给出更坏的结果。
- 更重要的是，如果评测协议本身偏弱，那么后续“训练更鲁棒了”的结论也会被系统性污染。

## 协议组成

- 优化器：APGD
- 优化技巧：progressive radius reduction，按 `2e -> 1.5e -> e` 的 red-epsilon 方式推进
- 单次攻击预算：每个 loss 统一运行 `300` iterations；附录分析表明这个长度已基本足够，且随机性较低。
- 组合方式：
  - accuracy：可逐图取最坏攻击
  - mIoU：因其是全验证集级别指标，精确全局最坏组合是组合爆炸问题；论文因此采用 greedy 近似搜索 per-image 组合

## 使用边界

- 它是强白盒协议，适合评测常规 adversarially trained segmentation models。
- 它不是完整的白盒 + 黑盒组合基准，因此对于依赖梯度掩蔽的防御，仍可能高估鲁棒性。
- 论文实证表明，对抗训练得到的模型通常不会因为这一点而被严重误判，但这一风险在方法论上仍然存在。

## 在本语料中的位置

- 代表论文：[[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]]
- 直接对比：[[concepts/SegPGD]]
- 上位主题：[[concepts/Segmentation Robustness Benchmark Protocol]]

## 实践建议

- 如果只做一次低成本 sanity check，优先在作者 released models 上跑 SEA，而不是先从零训练 PIR-AT。
- 写实验时要把 `red-epsilon APGD` 和 `SEA` 分开描述，前者是 optimizer / schedule，后者是 protocol。
- 自己复现时重点盯四个容易变弱的细节：background class、step size tuning、mIoU 的 greedy worst-case 逻辑、best-iterate selection。
- 报告 segmentation robustness 时，不要只报某一个攻击下的 accuracy，至少要同时给出 accuracy / mIoU 和攻击预算。
- 如果只能跑一个强基线，优先保留 SEA，而不是单独保留某个 loss。
- 如果评测对象可能有梯度问题，SEA 应与黑盒攻击或迁移攻击配合使用。

## 关联链接

- [[concepts/Prior-informed Robust Adversarial Training (PIR-AT)]]
- [[synthesis/Reliability and Benchmarking for Robust Segmentation]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
