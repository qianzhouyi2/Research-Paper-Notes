---
title: "Towards Semantically Stealthy Adversarial Attacks Against Segmentation Models"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/15_Towards_Semantically_Stealthy_Adversarial_Attacks_Against_Segmentation/2104.01732.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2104.01732
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-3 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Towards Semantically Stealthy Adversarial Attacks Against Segmentation Models

## Reading Status

- Batch: 3 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2021
- Venue: WACV
- Verification source: arxiv
- Verified title: Semantically Stealthy Adversarial Attacks against Segmentation Models

## Problem and Method (From Extracted Evidence)

- Evidence summary: Segmentation models have been found to be vulnerable to targeted and non-targeted adversarial attacks. However, the resulting segmentation outputs are often so damaged that it is easy to spot an attack.
- Method hint: In this paper, we propose semantically stealthy adversarial attacks which can manipulate targeted labels while preserving non-targeted labels at the same time. One challenge is making semantically meaningful manipulations across datasets and models.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related work
- Universal adversarial attacks
- Image-dependent adversarial attacks
- Semantic adversarial attacks
- Physical adversarial attacks
- Our Approach
- Problem definition

## Extracted Themes

- Theme tags pending deeper full-text parsing. ^[ambiguous]

## Experimental Context

- Dataset mentions in extracted text: Cityscapes
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Zhenhua Chen]]
- [[entities/Chuhua Wang]]
- [[entities/David J. Crandall]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2104.01732
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

