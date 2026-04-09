---
title: "Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/18_Certified_Defences_Against_Adversarial_Patch_Attacks_on_Semantic_Segme/2209.05980.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2209.05980
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-3 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation

## Reading Status

- Batch: 3 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2022
- Venue: ICLR
- Verification source: arxiv
- Verified title: Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation

## Problem and Method (From Extracted Evidence)

- Evidence summary: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing , the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model.
- Method hint: Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing , any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related Work
- Problem Setup
- Semantic Segmentation
- Threat model
- Defence objective
- Demasked Smoothing
- Input masking

## Extracted Themes

- Patch attack or defense
- Randomized smoothing / certification
- Certified robustness guarantees
- Adversarial training strategy

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Maksym Yatsura]]
- [[entities/Kaspar Sakmann]]
- [[entities/N. Grace Hua]]
- [[entities/Matthias Hein]]
- [[entities/Jan Hendrik Metzen]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2209.05980
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

