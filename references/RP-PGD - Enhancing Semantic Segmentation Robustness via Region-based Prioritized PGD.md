---
title: "RP-PGD: Enhancing Semantic Segmentation Robustness via Region-based Prioritized PGD"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/27_RP_PGD_Enhancing_Semantic_Segmentation_Robustness_via_Region_based_Pri/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.1609/aaai.v39i10.33122
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# RP-PGD: Enhancing Semantic Segmentation Robustness via Region-based Prioritized PGD

## 阅读状态

- 批次：4 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2025
- 发表 venue：AAAI
- 核验来源：crossref
- 核验标题：RP-PGD: Enhancing Semantic Segmentation Robustness via Region-based Prioritized PGD

## 问题与方法（基于抽取证据）

- 证据摘要：Adversarial attack and defense have been extensively ex- plored in classiﬁcation tasks, but their study in semantic seg- mentation remains limited. Moreover, current attacks fail to act as strong underlying attacks for adversarial training (AT), making it difﬁcult to achieve segmentation robustness against strong attacks.
- 方法线索：In this paper, we presentRP-PGD, a novel Region-and-Prototype based Projected Gradient Descent at- tack tailored to fool segmentation models. In particular, we propose a region-based attack, which leverages a spatial- temporal way to separate the pixels into three disjoint re- gions, and highlights the attack on the crucial True Region and Boundary Region.

## 本地论文结构证据

- 本轮未能从本地源格式稳定抽取章节结构。^[ambiguous]

## 抽取主题

- Adversarial training strategy

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：pdf

## 关联作者实体

- 暂无关联作者实体。^[ambiguous]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://doi.org/10.1609/aaai.v39i10.33122
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
