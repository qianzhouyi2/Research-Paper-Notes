---
title: "Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/22_Towards_Robust_Semantic_Segmentation_against_Patch_based_Attack_via_At/2401.01750.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2401.01750
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-4 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement

## Reading Status

- Batch: 4 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2024
- Venue: IJCV
- Verification source: arxiv
- Verified title: Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement

## Problem and Method (From Extracted Evidence)

- Evidence summary: The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones.
- Method hint: However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related Work
- Semantic Segmentation
- Patch-based Adversarial Attack
- Robustness of Vision Transformer
- Analyses of Receptive Field and Robustness
- Receptive Field
- Analysis

## Extracted Themes

- Patch attack or defense
- Attention refinement / regularization

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

- Primary verification link: https://arxiv.org/abs/2401.01750
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

