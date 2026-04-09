---
title: "Towards Robust General Medical Image Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/13_Towards_Robust_General_Medical_Image_Segmentation/2107.04263.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2107.04263
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-2 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Towards Robust General Medical Image Segmentation

## Reading Status

- Batch: 2 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2021
- Venue: MICCAI
- Verification source: arxiv
- Verified title: Towards Robust General Medical Image Segmentation

## Problem and Method (From Extracted Evidence)

- Evidence summary: The reliability of Deep Learning systems depends on their accuracy but also on their robustness against adversarial perturbations to the input data. Several attacks and defenses have been proposed to improve the performance of Deep Neural Networks under the presence of adversarial noise in the natural image domain.
- Method hint: However, robustness in computer-aided diagnosis for volumetric data has only been explored for specific tasks and with limited attacks. We propose a new framework to assess the robustness of general medical image segmentation systems.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Methodology
- Adversarial robustness
- Generic Medical Segmentation
- Experiments
- Adversarial robustness assessment
- General medical segmentation
- Results

## Extracted Themes

- Medical image segmentation
- Adversarial training strategy
- Benchmark protocol design
- Reliability-focused evaluation

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Laura Daza]]
- [[entities/Juan C. Perez]]
- [[entities/Pablo Arbelaez]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2107.04263
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

