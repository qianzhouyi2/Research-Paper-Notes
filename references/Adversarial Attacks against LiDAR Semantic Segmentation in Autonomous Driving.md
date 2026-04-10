---
title: "Adversarial Attacks against LiDAR Semantic Segmentation in Autonomous Driving"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/12_Adversarial_Attacks_against_LiDAR_Semantic_Segmentation_in_Autonomous_/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.1145/3485730.3485935
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Adversarial Attacks against LiDAR Semantic Segmentation in Autonomous Driving

## 阅读状态

- 批次：2 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2021
- 发表 venue：SenSys
- 核验来源：crossref
- 核验标题：Adversarial Attacks against LiDAR Semantic Segmentation in Autonomous Driving

## 问题与方法（基于抽取证据）

- 证据摘要：Today, most autonomous vehicles (AVs) rely on LiDAR (Light De- tection and Ranging) perception to acquire accurate information about their immediate surroundings. In LiDAR-based perception systems, semantic segmentation plays a critical role as it can divide LiDAR point clouds into meaningful regions according to human perception and provide AVs with semantic understanding of the driving environments.
- 方法线索：However, an implicit assumption for existing semantic segmentation models is that they are performed in a reli- able and secure environment, which may not be true in practice. In this paper, we investigate adversarial attacks against LiDAR seman- tic segmentation in autonomous driving.

## 本地论文结构证据

- 本轮未能从本地源格式稳定抽取章节结构。^[ambiguous]

## 抽取主题

- LiDAR point-cloud segmentation

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：pdf

## 关联作者实体

- [[entities/Yi Zhu]]
- [[entities/Chenglin Miao]]
- [[entities/Foad Hajiaghajani]]
- [[entities/Mengdi Huai]]
- [[entities/Lu Su]]
- [[entities/Chunming Qiao]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://doi.org/10.1145/3485730.3485935
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
