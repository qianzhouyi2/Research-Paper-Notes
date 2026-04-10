---
title: "Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/05_Benchmarking_the_Robustness_of_Semantic_Segmentation_Models/1908.05005.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/1908.05005
  - https://doi.org/10.1007/s11263-020-01383-2
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)

## 阅读状态

- 批次：1 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2021
- 发表 venue：IJCV
- 核验来源：crossref
- 核验标题：Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)

## 问题与方法（基于抽取证据）

- 证据摘要：When designing a semantic segmentation module for a practical application, such as autonomous driving, it is crucial to understand the robustness of the module with respect to a wide range of image corruptions. While there are recent robustness studies for full-image classification, we are the first to present an exhaustive study for semantic segmentation, based on the state-of-the-art model DeepLabv3 + .
- 方法线索：To increase the realism of our study, we utilize almost 400,000 images generated from Cityscapes, PASCAL VOC 2012, and ADE20K. Based on the benchmark study, we gain several new insights.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Related Work
- Image Corruption Models
- ImageNet-C
- Additional Image Corruptions
- Models
- DeepLabv3$+$
- Architectural Ablations

## 抽取主题

- Benchmark protocol design

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Christoph Kamann]]
- [[entities/Carsten Rother]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/1908.05005
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
