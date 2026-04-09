---
title: "Evaluating the Robustness of Semantic Segmentation in Real-World Adversarial Patch Attacks"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/14_Evaluating_the_Robustness_of_Semantic_Segmentation_in_Real_World_Adver/2108.06179.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2108.06179
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-2 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Evaluating the Robustness of Semantic Segmentation in Real-World Adversarial Patch Attacks

## Reading Status

- Batch: 2 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2021
- Venue: WACV
- Verification source: arxiv
- Verified title: Evaluating the Robustness of Semantic Segmentation for Autonomous Driving against Real-World Adversarial Patch Attacks

## Problem and Method (From Extracted Evidence)

- Evidence summary: Deep learning and convolutional neural networks allow achieving impressive performance in computer vision tasks, such as object detection and semantic segmentation (SS). However, recent studies have shown evident weaknesses of such models against adversarial perturbations.
- Method hint: In a real-world scenario instead, like autonomous driving, more attention should be devoted to real-world adversarial examples (RWAEs), which are physical objects (e.g., billboards and printable patches) optimized to be adversarial to the entire perception pipeline. This paper presents an in-depth evaluation of the robustness of popular SS models by testing the effects of both digital and real-world adversarial patches.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Experimental results
- Experimental setup
- EOT-based patches on Cityscapes
- Scene-specific patches on CARLA
- Real-world patches
- Introduction
- This paper
- Related Work

## Extracted Themes

- Patch attack or defense
- Attention refinement / regularization

## Experimental Context

- Dataset mentions in extracted text: Cityscapes
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Federico Nesti]]
- [[entities/Giulio Rossolini]]
- [[entities/Saasha Nair]]
- [[entities/Alessandro Biondi]]
- [[entities/Giorgio Buttazzo]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2108.06179
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

