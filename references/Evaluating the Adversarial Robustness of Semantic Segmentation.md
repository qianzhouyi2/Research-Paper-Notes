---
title: "Evaluating the Adversarial Robustness of Semantic Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/24_Evaluating_the_Adversarial_Robustness_of_Semantic_Segmentation/2306.14217.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2306.14217
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Evaluating the Adversarial Robustness of Semantic Segmentation

## 阅读状态

- 批次：4 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2023
- 发表 venue：ECCV
- 核验来源：arxiv
- 核验标题：Evaluating the Adversarial Robustness of Semantic Segmentation

## 问题与方法（基于抽取证据）

- 证据摘要：Achieving robustness against adversarial input perturbation is an important and intriguing problem in machine learning. In the area of semantic image segmentation, a number of adversarial training approaches have been proposed as a defense against adversarial perturbation, but the methodology of evaluating the robustness of the models is still lacking, compared to image classification.
- 方法线索：Here, we demonstrate that, just like in image classification, it is important to evaluate the models over several different and hard attacks. We propose a set of gradient based iterative attacks and show that it is essential to perform a large number of iterations.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Acknowledgements
- Investigated Models
- Robustness to Bounded Attacks
- Cosine Internal Representation Attacks
- Attacking Segmentation Networks
- Attacking the Internal Representations
- CIRA+: a Hybrid Attack
- CIRA vs. CIRA+

## 抽取主题

- Adversarial training strategy

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

- 主核验链接：https://arxiv.org/abs/2306.14217
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
