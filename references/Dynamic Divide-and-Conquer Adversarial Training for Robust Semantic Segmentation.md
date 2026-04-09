---
title: "Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/11_Dynamic_Divide_and_Conquer_Adversarial_Training_for_Robust_Semantic_Se/2003.06555.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2003.06555
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-2 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation

## Reading Status

- Batch: 2 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2020
- Venue: ICCV
- Verification source: arxiv
- Verified title: Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation

## Problem and Method (From Extracted Evidence)

- Evidence summary: Adversarial training is promising for improving robustness of deep neural networks towards adversarial perturbations, especially on the classification task. The effect of this type of training on semantic segmentation, contrarily, just commences.
- Method hint: We make the initial attempt to explore the defense strategy on semantic segmentation by formulating a general adversarial training procedure that can perform decently on both adversarial and clean samples. We propose a dynamic divide-and-conquer adversarial training (DDC-AT) strategy to enhance the defense effect, by setting additional branches in the target model during training, and dealing with pixels with diverse properties towards adversarial perturbation.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related Work
- Standard Adversarial Attack
- Standard Adversarial Training
- DDC-AT
- Divide-and-Conquer Procedure
- Dynamical Division and Implementation
- Overall Loss Function

## Extracted Themes

- Black-box attack setting
- Adversarial training strategy

## Experimental Context

- Dataset mentions in extracted text: Cityscapes
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Xiaogang Xu]]
- [[entities/Hengshuang Zhao]]
- [[entities/Jiaya Jia]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2003.06555
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

