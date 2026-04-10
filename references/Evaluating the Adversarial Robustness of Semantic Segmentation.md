---
title: "Evaluating the Adversarial Robustness of Semantic Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/24_Evaluating_the_Adversarial_Robustness_of_Semantic_Segmentation/2306.14217.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2306.14217
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-4 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Evaluating the Adversarial Robustness of Semantic Segmentation

## Reading Status

- Batch: 4 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2023
- Venue: arXiv preprint (with related ECCV 2024 follow-up)
- Verification source: arxiv+ecva
- Verified title: On Evaluating the Adversarial Robustness of Semantic Segmentation Models

## Problem and Method (From Extracted Evidence)

- Evidence summary: Achieving robustness against adversarial input perturbation is an important and intriguing problem in machine learning. In the area of semantic image segmentation, a number of adversarial training approaches have been proposed as a defense against adversarial perturbation, but the methodology of evaluating the robustness of the models is still lacking, compared to image classification.
- Method hint: Here, we demonstrate that, just like in image classification, it is important to evaluate the models over several different and hard attacks. We propose a set of gradient based iterative attacks and show that it is essential to perform a large number of iterations.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Acknowledgements
- Investigated Models
- Robustness to Bounded Attacks
- Cosine Internal Representation Attacks
- Attacking Segmentation Networks
- Attacking the Internal Representations
- CIRA+: a Hybrid Attack
- CIRA vs. CIRA+

## Extracted Themes

- Adversarial training strategy

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- Verified authors (arXiv metadata): Levente Halmosi; Mark Jelasity.

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2306.14217
- Related ECCV 2024 follow-up line: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10111_ECCV_2024_paper.php
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

