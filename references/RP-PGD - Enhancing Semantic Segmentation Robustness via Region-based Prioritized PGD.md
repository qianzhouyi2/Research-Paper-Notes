---
title: "RP-PGD: Enhancing Semantic Segmentation Robustness via Region-based Prioritized PGD"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/27_RP_PGD_Enhancing_Semantic_Segmentation_Robustness_via_Region_based_Pri/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.1609/aaai.v39i10.33122
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-4 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# RP-PGD: Enhancing Semantic Segmentation Robustness via Region-based Prioritized PGD

## Reading Status

- Batch: 4 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2025
- Venue: AAAI
- Verification source: crossref
- Verified title: RP-PGD: Boosting Segmentation Robustness with a Region-and-Prototype Based Adversarial Attack

## Problem and Method (From Extracted Evidence)

- Evidence summary: Adversarial attack and defense have been extensively ex- plored in classiﬁcation tasks, but their study in semantic seg- mentation remains limited. Moreover, current attacks fail to act as strong underlying attacks for adversarial training (AT), making it difﬁcult to achieve segmentation robustness against strong attacks.
- Method hint: In this paper, we presentRP-PGD, a novel Region-and-Prototype based Projected Gradient Descent at- tack tailored to fool segmentation models. In particular, we propose a region-based attack, which leverages a spatial- temporal way to separate the pixels into three disjoint re- gions, and highlights the attack on the crucial True Region and Boundary Region.

## Local Paper Structure Evidence

- Structure extraction unavailable from local source format in this pass. ^[ambiguous]

## Extracted Themes

- Adversarial training strategy

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: pdf

## Linked Author Entities

- Verified authors (Crossref metadata): Yuxuan Zhang; Zhenbo Shi; Shuchang Wang; Wei Yang; Shaowei Wang; Yinxing Xue.

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://doi.org/10.1609/aaai.v39i10.33122
- Secondary metadata link: https://dblp.org/rec/conf/aaai/ZhangSW00X25
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

