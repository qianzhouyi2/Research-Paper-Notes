---
title: "Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/25_Towards_Reliable_Evaluation_and_Fast_Training_of_Robust_Semantic_Segme/2306.12941.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2306.12941
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-4 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models

## Reading Status

- Batch: 4 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2023
- Venue: ECCV
- Verification source: arxiv
- Verified title: Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models

## Problem and Method (From Extracted Evidence)

- Evidence summary: Adversarial robustness has been studied extensively in image classification, especially for the _ -threat model, but significantly less so for related tasks such as object detection and semantic segmentation, where attacks turn out to be a much harder optimization problem than for image classification. We propose several problem-specific novel attacks minimizing different metrics in accuracy and mIoU.
- Method hint: The ensemble of our attacks, , shows that existing attacks severely overestimate the robustness of semantic segmentation models. Surprisingly, existing attempts of adversarial training for semantic segmentation models turn out to be weak or even completely non-robust.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related Work
- Adversarial Attacks for Semantic Segmentation
- Setup
- How to efficiently attack
- Why do attacks on semantic segmentation require new loss functions compared to image segmentation?
- Novel attacks on semantic segmentation
- Optimization techniques for adversarial attacks on semantic segmentation

## Extracted Themes

- Adversarial training strategy

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Matthias Hein]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2306.12941
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

