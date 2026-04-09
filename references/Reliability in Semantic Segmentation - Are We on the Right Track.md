---
title: "Reliability in Semantic Segmentation: Are We on the Right Track?"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/21_Reliability_in_Semantic_Segmentation_Are_We_on_the_Right_Track/2303.11298.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2303.11298
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-3 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Reliability in Semantic Segmentation: Are We on the Right Track?

## Reading Status

- Batch: 3 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2023
- Venue: CVPR
- Verification source: arxiv
- Verified title: Reliability in Semantic Segmentation: Are We on the Right Track?

## Problem and Method (From Extracted Evidence)

- Evidence summary: Motivated by the increasing popularity of transformers in computer vision, in recent times there has been a rapid development of novel architectures. While in-domain performance follows a constant, upward trend, properties like robustness or uncertainty estimation are less explored---leaving doubts about advances in model reliability .
- Method hint: Studies along these axes exist, but they are mainly limited to classification models. In contrast, we carry out a study on semantic segmentation, a relevant task for many real-world applications where model reliability is paramount.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related work
- Ablation of calibration metrics
- Ablation of number of pixels for calibration
- Ablation of confidence score: max probability  entropy
- Ablation number of clusters
- Visualization of cluster samples
- Subset calibration

## Extracted Themes

- Reliability-focused evaluation
- Uncertainty-driven detection

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Philip Torr]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2303.11298
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

