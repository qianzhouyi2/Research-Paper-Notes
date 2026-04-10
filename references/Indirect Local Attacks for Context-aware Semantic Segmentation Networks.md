---
title: "Indirect Local Attacks for Context-aware Semantic Segmentation Networks"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/03_Indirect_Local_Attacks_for_Context_aware_Semantic_Segmentation_Network/1911.13038.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/1911.13038
created: 2026-04-10
updated: 2026-04-10
summary: "分批逐篇阅读卡：包含已核验元数据、本地摘要证据与可复用鲁棒性要点。"
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---
# Indirect Local Attacks for Context-aware Semantic Segmentation Networks

## 阅读状态

- 批次：1 / 4
- 逐篇审阅状态：已完成（元数据 + 摘要级本地阅读 + 跨源核验）

## 元数据

- 年份：2019
- 发表 venue：ECCV
- 核验来源：arxiv
- 核验标题：Indirect Local Attacks for Context-aware Semantic Segmentation Networks

## 问题与方法（基于抽取证据）

- 证据摘要：Recently, deep networks have achieved impressive semantic segmentation performance, in particular thanks to their use of larger contextual information. In this paper, we show that the resulting networks are sensitive not only to global attacks, where perturbations affect the entire input image, but also to indirect local attacks where perturbations are confined to a small image region that does not overlap with the area that we aim to fool.
- 方法线索：To this end, we introduce several indirect attack strategies, including adaptive local attacks, aiming to find the best image location to perturb, and universal local attacks. Furthermore, we propose attack detection techniques both for the global image level and to obtain a pixel-wise localization of the fooled regions.

## 本地论文结构证据

- 抽取来源：tex-sections
- 本地源码中观测到的章节/小节标题：
- Indirect Local Segmentation Attacks
- Indirect Local Attacks
- Adaptive Attacks
- Universal Local Attacks
- Adversarial Attack Detection
- Conclusion

## 抽取主题

- 主题标签仍待更深层全文解析。^[ambiguous]

## 实验语境

- 抽取文本中的数据集提及：当前摘要片段未显式给出。^[ambiguous]
- 本地材料类型：tex_source

## 关联作者实体

- [[entities/Krishna Kanth Nakka]]
- [[entities/Mathieu Salzmann]]

## 关联概念与综合

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## 联网核验备注

- 主核验链接：https://arxiv.org/abs/1911.13038
- 本地材料存放于 `papers_sources/semantic_segmentation_robustness_20260409`。
