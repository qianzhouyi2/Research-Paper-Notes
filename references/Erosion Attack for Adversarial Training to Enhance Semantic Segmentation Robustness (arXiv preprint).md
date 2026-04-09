---
title: "Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness (arXiv preprint)"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/28_Erosion_Attack_for_Adversarial_Training_to_Enhance_Semantic_Segmentati/2601.14950.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2601.14950
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-4 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness (arXiv preprint)

## Reading Status

- Batch: 4 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2026
- Venue: arXiv
- Verification source: arxiv-local
- Verified title: Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness

## Problem and Method (From Extracted Evidence)

- Evidence summary: Existing segmentation models exhibit significant vulnerability to adversarial attacks. To improve robustness, adversarial training incorporates adversarial examples into model training.
- Method hint: However, existing attack methods consider only global semantic information and ignore contextual semantic relationships within the samples, limiting the effectiveness of adversarial training. To address this issue, we propose EroSeg-AT, a vulnerability-aware adversarial training framework that leverages EroSeg to generate adversarial examples.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Limitation and Conclusion
- Acknowledgements
- Related Works
- Semantic Segmentation Models
- Adversarial Training
- Methodology
- Motivation
- EroSeg: A Complete Illustration

## Extracted Themes

- Adversarial training strategy

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

- Primary verification link: https://arxiv.org/abs/2601.14950
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.
- Note: arXiv API rate-limit prevented direct query in this session; local arXiv id and downloaded source are used as primary evidence. ^[ambiguous]
