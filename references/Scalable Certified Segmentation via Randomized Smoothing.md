---
title: "Scalable Certified Segmentation via Randomized Smoothing"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/10_Scalable_Certified_Segmentation_via_Randomized_Smoothing/2107.00228.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2107.00228
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Scalable Certified Segmentation via Randomized Smoothing

## 阅读状态

- 批次：2 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2021
- 发表 venue：ICML
- 核验来源：arxiv
- 核验标题：Scalable Certified Segmentation via Randomized Smoothing

## 问题与方法（基于抽取证据）

- 证据摘要：We present a new certiﬁcation method for image and point cloud segmentation based on random- ized smoothing. The method leverages a novel scalable algorithm for prediction and certiﬁcation that correctly accounts for multiple testing, nec- essary for ensuring statistical guarantees.
- 方法线索：The key to our approach is reliance on established multiple-testing correction mechanisms as well as the ability to abstain from classifying single pixels or points while still robustly segmenting the overall input. Our experimental evaluation on synthetic data and challenging datasets, such as Pascal Context, Cityscapes, and ShapeNet, shows that our algorithm can achieve, for the ﬁrst time, competitive accuracy and certiﬁcation guarantees on real-world segmentation tasks.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Related Work
- Randomized Smoothing for Classification
- Conclusion
- Experimental Evaluation
- Toy Data
- Semantic Image Segmentation
- Pointcloud Part Segmentation

## 抽取主题

- Randomized smoothing / certification
- Certified robustness guarantees

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Marc Fischer]]
- [[entities/Maximilian Baader]]
- [[entities/Martin Vechev]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2107.00228
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
