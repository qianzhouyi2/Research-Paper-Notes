---
title: "Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/06_Increasing_the_Robustness_of_Semantic_Segmentation_Models_with_Paintin/2010.05495.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2010.05495
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers

## 阅读状态

- 批次：1 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2020
- 发表 venue：ECCV
- 核验来源：arxiv
- 核验标题：Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers

## 问题与方法（基于抽取证据）

- 证据摘要：For safety-critical applications such as autonomous driving, CNNs have to be robust with respect to unavoidable image corruptions, such as image noise. While previous works addressed the task of robust prediction in the context of full-image classification, we consider it for dense semantic segmentation.
- 方法线索：We build upon an insight from image classification that output robustness can be improved by increasing the network-bias towards object shapes. We present a new training schema that increases this shape bias.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Related Work
- Training Schema: Painting-by-Numbers
- Experimental Evaluation and Validation
- Implementation Details
- Results on Cityscapes
- Understanding Painting-by-Numbers
- Conclusions

## 抽取主题

- Adversarial training strategy
- Painting-by-numbers prior

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Christoph Kamann]]
- [[entities/Burkhard Gussefeld]]
- [[entities/Robin Hutmacher]]
- [[entities/Jan Hendrik Metzen]]
- [[entities/Carsten Rother]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2010.05495
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
