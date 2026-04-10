---
title: "Improved Noise and Attack Robustness of Semantic Segmentation Models with Self-Supervised Depth Estimation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/07_Improved_Noise_and_Attack_Robustness_of_Semantic_Segmentation_Models_w/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.1109/cvprw50498.2020.00168
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Improved Noise and Attack Robustness of Semantic Segmentation Models with Self-Supervised Depth Estimation

## 阅读状态

- 批次：1 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2020
- 发表 venue：CVPRW
- 核验来源：crossref
- 核验标题：Improved Noise and Attack Robustness of Semantic Segmentation Models with Self-Supervised Depth Estimation

## 问题与方法（基于抽取证据）

- 证据摘要：While current approaches for neural network training of- ten aim at improving performance, less focus is put on train- ing methods aiming at robustness towards varying noise conditions or directed attacks by adversarial examples. In this paper, we propose to improve robustness by a multi-task training, which extends supervised semantic segmentation by a self-supervised monocular depth estimation on unla- beled videos.
- 方法线索：This additional task is only performed dur- ing training to improve the semantic segmentation model’s robustness at test time under several input perturbations. Moreover, we even ﬁnd that our joint training approach also improves the performance of the model on the original (su- pervised) semantic segmentation task.

## 本地论文结构证据

- 本轮未能从本地源格式稳定抽取章节结构。^[ambiguous]

## 抽取主题

- Adversarial training strategy

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：pdf

## 关联作者实体

- [[entities/Marvin Klingner]]
- [[entities/Andreas Bar]]
- [[entities/Tim Fingscheidt]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://doi.org/10.1109/cvprw50498.2020.00168
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
