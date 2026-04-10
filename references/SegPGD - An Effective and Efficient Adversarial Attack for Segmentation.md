---
title: "SegPGD: An Effective and Efficient Adversarial Attack for Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/16_SegPGD_An_Effective_and_Efficient_Adversarial_Attack_for_Segmentation/2207.12391.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2207.12391
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# SegPGD: An Effective and Efficient Adversarial Attack for Segmentation

## 阅读状态

- 批次：3 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2022
- 发表 venue：ECCV
- 核验来源：arxiv
- 核验标题：SegPGD: An Effective and Efficient Adversarial Attack for Segmentation

## 问题与方法（基于抽取证据）

- 证据摘要：Deep neural network-based image classifications are vulnerable to adversarial perturbations. The image classifications can be easily fooled by adding artificial small and imperceptible perturbations to input images.
- 方法线索：As one of the most effective defense strategies, adversarial training was proposed to address the vulnerability of classification models, where the adversarial examples are created and injected into training data during training. The attack and defense of classification models have been intensively studied in past years.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Related Work
- SegPGD for Evaluating and Boosting Segmentation
- SegPGD: An Effective and Efficient Segmentation Attack
- Convergence Analysis of SegPGD
- Segmentation Adversarial Training with SegPGD
- Experiment
- Experimental Setting

## 抽取主题

- Adversarial training strategy
- Attention refinement / regularization
- SegPGD family attack/training

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Jindong Gu]]
- [[entities/Hengshuang Zhao]]
- [[entities/Volker Tresp]]
- [[entities/Philip Torr]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2207.12391
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
