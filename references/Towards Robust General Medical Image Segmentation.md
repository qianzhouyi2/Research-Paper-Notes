---
title: "Towards Robust General Medical Image Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/13_Towards_Robust_General_Medical_Image_Segmentation/2107.04263.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2107.04263
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Towards Robust General Medical Image Segmentation

## 阅读状态

- 批次：2 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2021
- 发表 venue：MICCAI
- 核验来源：arxiv
- 核验标题：Towards Robust General Medical Image Segmentation

## 问题与方法（基于抽取证据）

- 证据摘要：The reliability of Deep Learning systems depends on their accuracy but also on their robustness against adversarial perturbations to the input data. Several attacks and defenses have been proposed to improve the performance of Deep Neural Networks under the presence of adversarial noise in the natural image domain.
- 方法线索：However, robustness in computer-aided diagnosis for volumetric data has only been explored for specific tasks and with limited attacks. We propose a new framework to assess the robustness of general medical image segmentation systems.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Methodology
- Adversarial robustness
- Generic Medical Segmentation
- Experiments
- Adversarial robustness assessment
- General medical segmentation
- Results

## 抽取主题

- Medical image segmentation
- Adversarial training strategy
- Benchmark protocol design
- Reliability-focused evaluation

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Laura Daza]]
- [[entities/Juan C. Perez]]
- [[entities/Pablo Arbelaez]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/2107.04263
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
