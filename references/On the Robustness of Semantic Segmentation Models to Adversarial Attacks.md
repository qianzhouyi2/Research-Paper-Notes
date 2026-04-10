---
title: "On the Robustness of Semantic Segmentation Models to Adversarial Attacks"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/08_On_the_Robustness_of_Semantic_Segmentation_Models_to_Adversarial_Attac/1711.09856.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/1711.09856
  - https://doi.org/10.1109/TPAMI.2019.2919707
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# On the Robustness of Semantic Segmentation Models to Adversarial Attacks

## 阅读状态

- 批次：2 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2020
- 发表 venue：TPAMI
- 核验来源：crossref
- 核验标题：On the Robustness of Semantic Segmentation Models to Adversarial Attacks

## 问题与方法（基于抽取证据）

- 证据摘要：Deep Neural Networks (DNNs) have demonstrated exceptional performance on most recognition tasks such as image classification and segmentation. However, they have also been shown to be vulnerable to adversarial examples.
- 方法线索：This phenomenon has recently attracted a lot of attention but it has not been extensively studied on multiple, large-scale datasets and structured prediction tasks such as semantic segmentation which often require more specialised networks with additional components such as CRFs, dilated convolutions, skip-connections and multiscale processing. In this paper, we present what to our knowledge is the first rigorous evaluation of adversarial attacks on modern semantic segmentation models, using two large-scale datasets.

## 本地论文结构证据

- 本轮未能从本地源格式稳定抽取章节结构。^[ambiguous]

## 抽取主题

- Benchmark protocol design
- Attention refinement / regularization

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Anurag Arnab]]
- [[entities/Ondrej Miksik]]
- [[entities/Philip H. S. Torr]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/1711.09856
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
