---
title: "Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/11_Dynamic_Divide_and_Conquer_Adversarial_Training_for_Robust_Semantic_Se/2003.06555.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2003.06555
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation

## 阅读状态

- 批次：2 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2020
- 发表 venue：ICCV
- 核验来源：arxiv
- 核验标题：Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation

## 问题与方法（基于抽取证据）

- 证据摘要：Adversarial training is promising for improving robustness of deep neural networks towards adversarial perturbations, especially on the classification task. The effect of this type of training on semantic segmentation, contrarily, just commences.
- 方法线索：We make the initial attempt to explore the defense strategy on semantic segmentation by formulating a general adversarial training procedure that can perform decently on both adversarial and clean samples. We propose a dynamic divide-and-conquer adversarial training (DDC-AT) strategy to enhance the defense effect, by setting additional branches in the target model during training, and dealing with pixels with diverse properties towards adversarial perturbation.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Related Work
- Standard Adversarial Attack
- Standard Adversarial Training
- DDC-AT
- Divide-and-Conquer Procedure
- Dynamical Division and Implementation
- Overall Loss Function

## 抽取主题

- Black-box attack setting
- Adversarial training strategy

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Xiaogang Xu]]
- [[entities/Hengshuang Zhao]]
- [[entities/Jiaya Jia]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2003.06555
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
