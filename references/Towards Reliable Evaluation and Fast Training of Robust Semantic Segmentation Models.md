---
title: "Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/25_Towards_Reliable_Evaluation_and_Fast_Training_of_Robust_Semantic_Segme/2306.12941.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2306.12941
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models

## 阅读状态

- 批次：4 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2023
- 发表 venue：ECCV
- 核验来源：arxiv
- 核验标题：Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models

## 问题与方法（基于抽取证据）

- 证据摘要：Adversarial robustness has been studied extensively in image classification, especially for the _ -threat model, but significantly less so for related tasks such as object detection and semantic segmentation, where attacks turn out to be a much harder optimization problem than for image classification. We propose several problem-specific novel attacks minimizing different metrics in accuracy and mIoU.
- 方法线索：The ensemble of our attacks, , shows that existing attacks severely overestimate the robustness of semantic segmentation models. Surprisingly, existing attempts of adversarial training for semantic segmentation models turn out to be weak or even completely non-robust.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Related Work
- Adversarial Attacks for Semantic Segmentation
- Setup
- How to efficiently attack
- Why do attacks on semantic segmentation require new loss functions compared to image segmentation?
- Novel attacks on semantic segmentation
- Optimization techniques for adversarial attacks on semantic segmentation

## 抽取主题

- Adversarial training strategy

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Matthias Hein]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2306.12941
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
