---
title: "Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/05_Benchmarking_the_Robustness_of_Semantic_Segmentation_Models/1908.05005.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/1908.05005
  - https://doi.org/10.1007/s11263-020-01383-2
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-1 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)

## Reading Status

- Batch: 1 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2021
- Venue: IJCV
- Verification source: crossref+arxiv
- Verified title: Benchmarking the Robustness of Semantic Segmentation Models with Respect to Common Corruptions

## Problem and Method (From Extracted Evidence)

- Evidence summary: When designing a semantic segmentation module for a practical application, such as autonomous driving, it is crucial to understand the robustness of the module with respect to a wide range of image corruptions. While there are recent robustness studies for full-image classification, we are the first to present an exhaustive study for semantic segmentation, based on the state-of-the-art model DeepLabv3 + .
- Method hint: To increase the realism of our study, we utilize almost 400,000 images generated from Cityscapes, PASCAL VOC 2012, and ADE20K. Based on the benchmark study, we gain several new insights.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related Work
- Image Corruption Models
- ImageNet-C
- Additional Image Corruptions
- Models
- DeepLabv3$+$
- Architectural Ablations

## Extracted Themes

- Benchmark protocol design

## Experimental Context

- Dataset mentions in extracted text: Cityscapes, ADE20K
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Christoph Kamann]]
- [[entities/Carsten Rother]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/1908.05005
- DOI verification link: https://doi.org/10.1007/s11263-020-01383-2
- Note: this is the journal extension line of the CVPR 2020 benchmark.
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

