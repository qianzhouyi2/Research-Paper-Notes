---
title: "Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - evaluation
  - efficiency
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/25_Towards_Reliable_Evaluation_and_Fast_Training_of_Robust_Semantic_Segme/metadata.json
  - papers_sources/semantic_segmentation_robustness_20260409/25_Towards_Reliable_Evaluation_and_Fast_Training_of_Robust_Semantic_Segme/2306.12941.tar
  - papers_sources/semantic_segmentation_robustness_20260409/25_Towards_Reliable_Evaluation_and_Fast_Training_of_Robust_Semantic_Segme/paper_resources/arxiv_2306.12941_translated.md
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2306.12941
created: 2026-04-10
updated: 2026-04-16
summary: "单篇摘要卡：这篇论文用 SEA 重新校准语义分割鲁棒评测，并用 PIR-AT 把鲁棒训练成本显著压低。"
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models

## 阅读状态

- 批次：4 / 4
- 逐篇审阅状态：已完成（元数据 + 本地全文抽取 + note / concept / entity 同步）

## 元数据

- 正式 venue：ECCV 2024
- 预印本锚点：`arXiv:2306.12941`（2023 年 6 月）
- 核验来源：本地 `metadata.json` + arXiv
- 作者：[[entities/Francesco Croce]]、[[entities/Naman D. Singh]]、[[entities/Matthias Hein]]

## 问题与方法（基于抽取证据）

- 核心问题：已有语义分割鲁棒评测不够强，导致很多“鲁棒模型”结论并不可靠；同时 segmentation adversarial training 的成本高到难以扩展。
- 方法摘要：
  - 攻击侧提出三类 segmentation-specific losses：`L_JS`、`L_MCE`、`L_MCE-Bal`
  - 优化侧使用 red-epsilon APGD
  - 评测侧把三次攻击集成为 [[concepts/Standardized Evaluation Attack (SEA) Protocol|Segmentation Ensemble Attack (SEA)]]
  - 训练侧提出 [[concepts/Prior-informed Robust Adversarial Training (PIR-AT)|PIR-AT]]，用鲁棒 ImageNet backbone 初始化 segmentation backbone
- 方法边界：SEA 是强白盒协议，但论文明确承认它还缺少黑盒补充，因此对梯度掩蔽防御仍可能保守不够。

## 关键证据

- 评测结论：
  - SEA 显示 SegPGD / CosPGD 会显著高估鲁棒性，尤其是在 mIoU 上。
  - DDC-AT 在更强评测下接近完全不鲁棒。
- 训练结论：
  - Pascal-Voc 上，`UPerNet + ConvNeXt-T, 5-step PIR-AT, 50 epochs` 在 `8/255` 下达到 `71.7%` robust accuracy，而 DDC-AT 为 `0.0%`。
  - Ade20K 上，`UPerNet + ConvNeXt-T, 5-step PIR-AT, 128 epochs` 在 `4/255` 下达到 `55.5%` robust accuracy。
  - Pascal-Voc 上 `2-step PIR-AT, 50 epochs` 已优于 `2-step AT, 300 epochs`；Ade20K 上 `32 epochs PIR-AT` 超过 `128 epochs AT`。

## 本地论文结构证据

- 抽取来源：tex-sections + translated markdown
- 本地源码中观测到的核心结构：
  - Introduction
  - Related Work
  - Adversarial Attacks for Semantic Segmentation
  - Novel attacks on semantic segmentation
  - Optimization techniques for adversarial attacks on semantic segmentation
  - Segmentation Ensemble Attack (SEA)
  - Adversarially Robust Segmentation Models
  - PIR-AT: robust models via robust initialization
  - Ablation study of AT vs PIR-AT

## 实验语境

- 数据集：Pascal-Voc、[[entities/ADE20K Dataset]]
- 模型：[[entities/PSPNet]]、UPerNet + ConvNeXt-T / S、Segmenter + ViT-S
- 指标：average pixel accuracy、mIoU
- 训练与评测关键点：
  - 训练统一使用 `epsilon = 4/255` 的 PGD adversarial training
  - 评测统一使用 `300` iterations 的 SEA
  - background class 被纳入训练和评测

## 关联概念与综合

- [[concepts/Standardized Evaluation Attack (SEA) Protocol|Segmentation Ensemble Attack (SEA) Protocol]]
- [[concepts/Prior-informed Robust Adversarial Training (PIR-AT)]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]
- [[synthesis/Reliability and Benchmarking for Robust Segmentation]]
- [[synthesis/Robust Training Strategies for Semantic Segmentation]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2306.12941
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`
- 时间说明：本页将 `arXiv:2306.12941` 视为预印本锚点，将 `ECCV 2024` 视为正式发表信息，避免年份字段混淆。
