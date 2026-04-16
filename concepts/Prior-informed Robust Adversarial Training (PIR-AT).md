---
title: "Prior-informed Robust Adversarial Training (PIR-AT)"
category: concept
tags:
  - concept
  - semantic-segmentation
  - adversarial-robustness
  - efficiency
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/25_Towards_Reliable_Evaluation_and_Fast_Training_of_Robust_Semantic_Segme/paper_resources/arxiv_2306.12941_translated.md
  - references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models.md
  - https://arxiv.org/abs/2306.12941
created: 2026-04-10
updated: 2026-04-16
summary: "PIR-AT：把 robust ImageNet initialization 引入 segmentation adversarial training 的 recipe；核心收益来自更好的鲁棒起点，而不是新的训练损失。"
provenance:
  extracted: 0.79
  inferred: 0.18
  ambiguous: 0.03
---

# Prior-informed Robust Adversarial Training (PIR-AT)

## 定义

PIR-AT 是一种用于语义分割鲁棒训练的初始化策略：不再从 clean ImageNet 预训练模型或随机参数出发，而是先用 `L_infty` 鲁棒的 ImageNet 分类器初始化 segmentation backbone，再在分割任务上做 adversarial training。本质上它是初始化策略和训练 recipe，不是新的训练目标函数。

## 核心做法

- backbone：用已经对抗训练过的 ImageNet 分类器初始化。
- decoder：仍随机初始化。
- 训练目标：仍然是标准的 segmentation adversarial training，论文里用的是 `epsilon = 4/255` 的 PGD + CE。
- 关键变化：不是改训练 loss，而是改“从哪里开始训练”。
- 重要区分：论文评测阶段使用更强的 `SEA / L_JS / L_MCE / L_MCE-Bal`，但训练阶段并没有直接把这些 loss 搬进训练。

## 为什么有效

- segmentation adversarial training 很贵，而且从 clean init 出发常常需要很多 epoch 才能进入有效鲁棒区间。
- robust backbone 已经带有更稳的局部纹理和判别表征，相当于把鲁棒特征预先迁移到 segmentation 任务。
- 在论文中，这个改动直接改变了训练成本曲线，支持“robust init > longer training”这条判断：
  - Pascal-Voc 上，`2-step PIR-AT, 50 epochs` 超过 `2-step AT, 300 epochs`
  - Ade20K 上，`32 epochs PIR-AT` 超过 `128 epochs AT`
- 这也说明一个容易误解的点：更强攻击 loss 不一定更适合作训练 loss；作者自己的做法仍是 `CE + PGD` 训练，只把 robust initialization 当成关键变量。

## 适用边界

- 适合：目标任务是 dense prediction，但 backbone 家族和分类预训练家族能够自然对接。
- 不适合直接照搬的情况：
  - 没有可用的 robust pretrained backbone
  - backbone / decoder 结构与预训练模型差异太大，无法自然对接预训练权重
  - 任务分布与 ImageNet 差得太远，迁移价值不足
- 额外限制：它的收益高度依赖可获得的 robust pretrained backbone 的质量、鲁棒半径和预训练输入设置。

## 在本语料中的位置

- 代表论文：[[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]]
- 关联主题：[[synthesis/Robust Training Strategies for Semantic Segmentation]]
- 关联评测协议：[[concepts/Standardized Evaluation Attack (SEA) Protocol|Segmentation Ensemble Attack (SEA) Protocol]]

## 实践建议

- 报告 PIR-AT 时，应把“robust initialization”单独列为实验变量，避免和攻击步数、训练 epoch 混在一起。
- 如果只复现一个最关键对照，优先做同架构、同步数、同 epoch 的 `AT vs PIR-AT`，这最容易看清真实增益来源。
- 如果复现实验，优先核对 backbone 的来源、鲁棒半径和输入分辨率，而不是先改 decoder 细节。
- 如果想做更强基线，可以先比较 `AT vs PIR-AT`，再比较 `2-step vs 5-step`，这样最容易看清它的真实增益来源。

## 关联链接

- [[concepts/Segmentation Robustness Benchmark Protocol]]
- [[entities/RobustBench]]
- [[synthesis/Reliability and Benchmarking for Robust Segmentation]]
