---
title: "Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/06_Increasing_the_Robustness_of_Semantic_Segmentation_Models_with_Paintin/2010.05495.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2010.05495
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-1 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers

## Reading Status

- Batch: 1 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2020
- Venue: ECCV
- Verification source: arxiv
- Verified title: Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers

## Problem and Method (From Extracted Evidence)

- Evidence summary: For safety-critical applications such as autonomous driving, CNNs have to be robust with respect to unavoidable image corruptions, such as image noise. While previous works addressed the task of robust prediction in the context of full-image classification, we consider it for dense semantic segmentation.
- Method hint: We build upon an insight from image classification that output robustness can be improved by increasing the network-bias towards object shapes. We present a new training schema that increases this shape bias.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related Work
- Training Schema: Painting-by-Numbers
- Experimental Evaluation and Validation
- Implementation Details
- Results on Cityscapes
- Understanding Painting-by-Numbers
- Conclusions

## Extracted Themes

- Adversarial training strategy
- Painting-by-numbers prior

## Experimental Context

- Dataset mentions in extracted text: Cityscapes
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Christoph Kamann]]
- [[entities/Burkhard Gussefeld]]
- [[entities/Robin Hutmacher]]
- [[entities/Jan Hendrik Metzen]]
- [[entities/Carsten Rother]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2010.05495
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

