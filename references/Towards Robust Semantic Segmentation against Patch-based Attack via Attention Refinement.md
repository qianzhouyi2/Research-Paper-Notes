---
title: "Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/22_Towards_Robust_Semantic_Segmentation_against_Patch_based_Attack_via_At/2401.01750.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2401.01750
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement

## 阅读状态

- 批次：4 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2024
- 发表 venue：IJCV
- 核验来源：arxiv
- 核验标题：Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement

## 问题与方法（基于抽取证据）

- 证据摘要：The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones.
- 方法线索：However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Introduction
- Related Work
- Semantic Segmentation
- Patch-based Adversarial Attack
- Robustness of Vision Transformer
- Analyses of Receptive Field and Robustness
- Receptive Field
- Analysis

## 抽取主题

- Patch attack or defense
- Attention refinement / regularization

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

- 主核验链接：https://arxiv.org/abs/2401.01750
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
