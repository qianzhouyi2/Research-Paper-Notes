---
title: "Proximal Splitting Adversarial Attack for Semantic Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/20_Proximal_Splitting_Adversarial_Attack_for_Semantic_Segmentation/2206.07179.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2206.07179
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Proximal Splitting Adversarial Attack for Semantic Segmentation

## 阅读状态

- 批次：3 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2023
- 发表 venue：CVPR
- 核验来源：arxiv
- 核验标题：Proximal Splitting Adversarial Attack for Semantic Segmentation

## 问题与方法（基于抽取证据）

- 证据摘要：Classification has been the focal point of research on adversarial attacks, but only a few works investigate methods suited to denser prediction tasks, such as semantic segmentation. The methods proposed in these works do not accurately solve the adversarial segmentation problem and, therefore, overestimate the size of the perturbations required to fool models.
- 方法线索：Here, we propose a white-box attack for these models based on a proximal splitting to produce adversarial perturbations with much smaller _ norms. Our attack can handle large numbers of constraints within a nonconvex minimization framework via an Augmented Lagrangian approach, coupled with adaptive constraint scaling and masking strategies.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Proposed Method
- Adaptive constraints strategies
- Proximal splitting
- Results
- Dense Adversary Generation attack
- Proof of Proposition 1
- ALMA prox attack algorithm

## 抽取主题

- Benchmark protocol design

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- 暂无关联作者实体。^[ambiguous]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2206.07179
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
