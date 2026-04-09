---
title: "Adversarial Attacks against LiDAR Semantic Segmentation in Autonomous Driving"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/12_Adversarial_Attacks_against_LiDAR_Semantic_Segmentation_in_Autonomous_/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.1145/3485730.3485935
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-2 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Adversarial Attacks against LiDAR Semantic Segmentation in Autonomous Driving

## Reading Status

- Batch: 2 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2021
- Venue: SenSys
- Verification source: crossref
- Verified title: Adversarial Attacks against LiDAR Semantic Segmentation in Autonomous Driving

## Problem and Method (From Extracted Evidence)

- Evidence summary: Today, most autonomous vehicles (AVs) rely on LiDAR (Light De- tection and Ranging) perception to acquire accurate information about their immediate surroundings. In LiDAR-based perception systems, semantic segmentation plays a critical role as it can divide LiDAR point clouds into meaningful regions according to human perception and provide AVs with semantic understanding of the driving environments.
- Method hint: However, an implicit assumption for existing semantic segmentation models is that they are performed in a reli- able and secure environment, which may not be true in practice. In this paper, we investigate adversarial attacks against LiDAR seman- tic segmentation in autonomous driving.

## Local Paper Structure Evidence

- Structure extraction unavailable from local source format in this pass. ^[ambiguous]

## Extracted Themes

- LiDAR point-cloud segmentation

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: pdf

## Linked Author Entities

- [[entities/Yi Zhu]]
- [[entities/Chenglin Miao]]
- [[entities/Foad Hajiaghajani]]
- [[entities/Mengdi Huai]]
- [[entities/Lu Su]]
- [[entities/Chunming Qiao]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://doi.org/10.1145/3485730.3485935
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

