---
title: "Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/18_Certified_Defences_Against_Adversarial_Patch_Attacks_on_Semantic_Segme/2209.05980.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2209.05980
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation

## 阅读状态

- 批次：3 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2022
- 发表 venue：ICLR
- 核验来源：arxiv
- 核验标题：Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation

## 问题与方法（基于抽取证据）

- 证据摘要：Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing , the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model.
- 方法线索：Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing , any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Related Work
- Problem Setup
- Semantic Segmentation
- Threat model
- Defence objective
- Demasked Smoothing
- Input masking

## 抽取主题

- Patch attack or defense
- Randomized smoothing / certification
- Certified robustness guarantees
- Adversarial training strategy

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Maksym Yatsura]]
- [[entities/Kaspar Sakmann]]
- [[entities/N. Grace Hua]]
- [[entities/Matthias Hein]]
- [[entities/Jan Hendrik Metzen]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2209.05980
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
