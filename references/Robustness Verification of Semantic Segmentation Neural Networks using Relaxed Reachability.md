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
summary: "Batch-2 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Robustness Verification of Semantic Segmentation Neural Networks using Relaxed Reachability

## Reading Status

- Batch: 2 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2021
- Venue: CAV
- Verification source: crossref
- Verified title: Robustness Verification of Semantic Segmentation Neural Networks Using Relaxed Reachability

## Problem and Method (From Extracted Evidence)

- Evidence summary: This paper introduces robustness veriﬁcation for semantic segmentation neural networks (in short, semantic segmentation networks [SSNs]), building on and extending recent approaches for robustness ver- iﬁcation of image classiﬁcation neural networks. Despite recent progress in developing veriﬁcation methods for speciﬁcations such as local adver- sarial robustness in deep neural networks (DNNs) in terms of scalability, precision, and applicability to diﬀerent network architectures, layers, and activation functions, robustness veriﬁcation of semantic segmentation has not yet been considered.
- Method hint: We address this limitation by developing and applying new robustness analysis methods for several segmentation neu- ral network architectures, speciﬁcally by addressing reachability anal- ysis of up-sampling layers, such as transposed convolution and dilated convolution. We consider several deﬁnitions of robustness for segmenta- tion, such as the percentage of pixels in the output that can be proven robust under diﬀerent adversarial perturbations, and a robust variant of intersection-over-union (IoU), the typical performance evaluation mea- sure for segmentation tasks.

## Local Paper Structure Evidence

- Structure extraction unavailable from local source format in this pass. ^[ambiguous]

## Extracted Themes

- Formal verification / reachability

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: pdf

## Linked Author Entities

- [[entities/Hoang-Dung Tran]]
- [[entities/Neelanjana Pal]]
- [[entities/Patrick Musau]]
- [[entities/Diego Manzanas Lopez]]
- [[entities/Nathaniel Hamilton]]
- [[entities/Xiaodong Yang]]
- [[entities/Stanley Bak]]
- [[entities/Taylor T. Johnson]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://doi.org/10.1007/978-3-030-81685-8_12
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

