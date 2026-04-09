---
title: "Proximal Splitting Adversarial Attack for Semantic Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/20_Proximal_Splitting_Adversarial_Attack_for_Semantic_Segmentation/2206.07179.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2206.07179
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-3 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Proximal Splitting Adversarial Attack for Semantic Segmentation

## Reading Status

- Batch: 3 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2022
- Venue: CVPR
- Verification source: arxiv
- Verified title: Proximal Splitting Adversarial Attacks for Semantic Segmentation

## Problem and Method (From Extracted Evidence)

- Evidence summary: Classification has been the focal point of research on adversarial attacks, but only a few works investigate methods suited to denser prediction tasks, such as semantic segmentation. The methods proposed in these works do not accurately solve the adversarial segmentation problem and, therefore, overestimate the size of the perturbations required to fool models.
- Method hint: Here, we propose a white-box attack for these models based on a proximal splitting to produce adversarial perturbations with much smaller _ norms. Our attack can handle large numbers of constraints within a nonconvex minimization framework via an Augmented Lagrangian approach, coupled with adaptive constraint scaling and masking strategies.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Proposed Method
- Adaptive constraints strategies
- Proximal splitting
- Results
- Dense Adversary Generation attack
- Proof of Proposition 1
- ALMA prox attack algorithm

## Extracted Themes

- Benchmark protocol design

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- Author entities pending for this paper. ^[ambiguous]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2206.07179
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

