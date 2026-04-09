---
title: "Scalable Certified Segmentation via Randomized Smoothing"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/10_Scalable_Certified_Segmentation_via_Randomized_Smoothing/2107.00228.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2107.00228
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-2 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Scalable Certified Segmentation via Randomized Smoothing

## Reading Status

- Batch: 2 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2021
- Venue: ICML
- Verification source: arxiv
- Verified title: Scalable Certified Segmentation via Randomized Smoothing

## Problem and Method (From Extracted Evidence)

- Evidence summary: We present a new certiﬁcation method for image and point cloud segmentation based on random- ized smoothing. The method leverages a novel scalable algorithm for prediction and certiﬁcation that correctly accounts for multiple testing, nec- essary for ensuring statistical guarantees.
- Method hint: The key to our approach is reliance on established multiple-testing correction mechanisms as well as the ability to abstain from classifying single pixels or points while still robustly segmenting the overall input. Our experimental evaluation on synthetic data and challenging datasets, such as Pascal Context, Cityscapes, and ShapeNet, shows that our algorithm can achieve, for the ﬁrst time, competitive accuracy and certiﬁcation guarantees on real-world segmentation tasks.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related Work
- Randomized Smoothing for Classification
- Conclusion
- Experimental Evaluation
- Toy Data
- Semantic Image Segmentation
- Pointcloud Part Segmentation

## Extracted Themes

- Randomized smoothing / certification
- Certified robustness guarantees

## Experimental Context

- Dataset mentions in extracted text: Cityscapes
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Marc Fischer]]
- [[entities/Maximilian Baader]]
- [[entities/Martin Vechev]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2107.00228
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

