---
title: "Improved Noise and Attack Robustness of Semantic Segmentation Models with Self-Supervised Depth Estimation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/07_Improved_Noise_and_Attack_Robustness_of_Semantic_Segmentation_Models_w/paper.pdf
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://doi.org/10.1109/cvprw50498.2020.00168
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-1 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Improved Noise and Attack Robustness of Semantic Segmentation Models with Self-Supervised Depth Estimation

## Reading Status

- Batch: 1 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2020
- Venue: CVPRW
- Verification source: crossref
- Verified title: Improved Noise and Attack Robustness for Semantic Segmentation by Using Multi-Task Training with Self-Supervised Depth Estimation

## Problem and Method (From Extracted Evidence)

- Evidence summary: While current approaches for neural network training of- ten aim at improving performance, less focus is put on train- ing methods aiming at robustness towards varying noise conditions or directed attacks by adversarial examples. In this paper, we propose to improve robustness by a multi-task training, which extends supervised semantic segmentation by a self-supervised monocular depth estimation on unla- beled videos.
- Method hint: This additional task is only performed dur- ing training to improve the semantic segmentation model’s robustness at test time under several input perturbations. Moreover, we even ﬁnd that our joint training approach also improves the performance of the model on the original (su- pervised) semantic segmentation task.

## Local Paper Structure Evidence

- Structure extraction unavailable from local source format in this pass. ^[ambiguous]

## Extracted Themes

- Adversarial training strategy

## Experimental Context

- Dataset mentions in extracted text: Cityscapes
- Local artifact type: pdf

## Linked Author Entities

- [[entities/Marvin Klingner]]
- [[entities/Andreas Bar]]
- [[entities/Tim Fingscheidt]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://doi.org/10.1109/cvprw50498.2020.00168
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

