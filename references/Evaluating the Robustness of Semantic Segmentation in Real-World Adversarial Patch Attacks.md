---
title: "Evaluating the Robustness of Semantic Segmentation in Real-World Adversarial Patch Attacks"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/14_Evaluating_the_Robustness_of_Semantic_Segmentation_in_Real_World_Adver/2108.06179.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2108.06179
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Evaluating the Robustness of Semantic Segmentation in Real-World Adversarial Patch Attacks

## 阅读状态

- 批次：2 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2021
- 发表 venue：WACV
- 核验来源：arxiv
- 核验标题：Evaluating the Robustness of Semantic Segmentation in Real-World Adversarial Patch Attacks

## 问题与方法（基于抽取证据）

- 证据摘要：Deep learning and convolutional neural networks allow achieving impressive performance in computer vision tasks, such as object detection and semantic segmentation (SS). However, recent studies have shown evident weaknesses of such models against adversarial perturbations.
- 方法线索：In a real-world scenario instead, like autonomous driving, more attention should be devoted to real-world adversarial examples (RWAEs), which are physical objects (e.g., billboards and printable patches) optimized to be adversarial to the entire perception pipeline. This paper presents an in-depth evaluation of the robustness of popular SS models by testing the effects of both digital and real-world adversarial patches.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Experimental results
- Experimental setup
- EOT-based patches on Cityscapes
- Scene-specific patches on CARLA
- Real-world patches
- Introduction
- This paper
- Related Work

## 抽取主题

- Patch attack or defense
- Attention refinement / regularization

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Federico Nesti]]
- [[entities/Giulio Rossolini]]
- [[entities/Saasha Nair]]
- [[entities/Alessandro Biondi]]
- [[entities/Giorgio Buttazzo]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2108.06179
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
