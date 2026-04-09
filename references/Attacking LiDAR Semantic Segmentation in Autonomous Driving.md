---
title: "Attacking LiDAR Semantic Segmentation in Autonomous Driving"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/17_Attacking_LiDAR_Semantic_Segmentation_in_Autonomous_Driving/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.14722/autosec.2022.23022
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-3 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Attacking LiDAR Semantic Segmentation in Autonomous Driving

## Reading Status

- Batch: 3 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2022
- Venue: NDSS
- Verification source: crossref
- Verified title: Demo: Attacking LiDAR Semantic Segmentation in Autonomous Driving

## Problem and Method (From Extracted Evidence)

- Evidence summary: —As a fundamental task in autonomous driving, LiDAR semantic segmentation aims to provide semantic un- derstanding of the driving environment. We demonstrate that existing LiDAR semantic segmentation models in autonomous driving systems can be easily fooled by placing some simple objects on the road, such as cardboard and traffic signs.
- Method hint: We show that this type of attack can hide a vehicle and change the road surface to road-side vegetation. T he development of autonomous vehicles (A Vs) has gained an increasing amount of momentum in recent years.

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

- Primary verification link: https://doi.org/10.14722/autosec.2022.23022
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

