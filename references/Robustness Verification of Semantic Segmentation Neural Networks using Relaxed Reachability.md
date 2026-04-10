---
title: "Robustness Verification of Semantic Segmentation Neural Networks using Relaxed Reachability"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/09_Robustness_Verification_of_Semantic_Segmentation_Neural_Networks_using/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.1007/978-3-030-81685-8_12
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Robustness Verification of Semantic Segmentation Neural Networks using Relaxed Reachability

## 阅读状态

- 批次：2 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2021
- 发表 venue：CAV
- 核验来源：crossref
- 核验标题：Robustness Verification of Semantic Segmentation Neural Networks using Relaxed Reachability

## 问题与方法（基于抽取证据）

- 证据摘要：This paper introduces robustness veriﬁcation for semantic segmentation neural networks (in short, semantic segmentation networks [SSNs]), building on and extending recent approaches for robustness ver- iﬁcation of image classiﬁcation neural networks. Despite recent progress in developing veriﬁcation methods for speciﬁcations such as local adver- sarial robustness in deep neural networks (DNNs) in terms of scalability, precision, and applicability to diﬀerent network architectures, layers, and activation functions, robustness veriﬁcation of semantic segmentation has not yet been considered.
- 方法线索：We address this limitation by developing and applying new robustness analysis methods for several segmentation neu- ral network architectures, speciﬁcally by addressing reachability anal- ysis of up-sampling layers, such as transposed convolution and dilated convolution. We consider several deﬁnitions of robustness for segmenta- tion, such as the percentage of pixels in the output that can be proven robust under diﬀerent adversarial perturbations, and a robust variant of intersection-over-union (IoU), the typical performance evaluation mea- sure for segmentation tasks.

## 本地论文结构证据

- 本轮未能从本地源格式稳定抽取章节结构。^[ambiguous]

## 抽取主题

- Formal verification / reachability

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：pdf

## 关联作者实体

- [[entities/Hoang-Dung Tran]]
- [[entities/Neelanjana Pal]]
- [[entities/Patrick Musau]]
- [[entities/Diego Manzanas Lopez]]
- [[entities/Nathaniel Hamilton]]
- [[entities/Xiaodong Yang]]
- [[entities/Stanley Bak]]
- [[entities/Taylor T. Johnson]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://doi.org/10.1007/978-3-030-81685-8_12
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
