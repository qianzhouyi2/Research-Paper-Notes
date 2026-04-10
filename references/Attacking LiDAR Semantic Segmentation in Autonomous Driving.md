---
title: "Attacking LiDAR Semantic Segmentation in Autonomous Driving"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/17_Attacking_LiDAR_Semantic_Segmentation_in_Autonomous_Driving/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.14722/autosec.2022.23022
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Attacking LiDAR Semantic Segmentation in Autonomous Driving

## 阅读状态

- 批次：3 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2022
- 发表 venue：NDSS
- 核验来源：crossref
- 核验标题：Attacking LiDAR Semantic Segmentation in Autonomous Driving

## 问题与方法（基于抽取证据）

- 证据摘要：As a fundamental task in autonomous driving, LiDAR semantic segmentation aims to provide semantic un- derstanding of the driving environment. We demonstrate that existing LiDAR semantic segmentation models in autonomous driving systems can be easily fooled by placing some simple objects on the road, such as cardboard and traffic signs.
- 方法线索：We show that this type of attack can hide a vehicle and change the road surface to road-side vegetation. T he development of autonomous vehicles (A Vs) has gained an increasing amount of momentum in recent years.

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

- 主核验链接：https://doi.org/10.14722/autosec.2022.23022
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
