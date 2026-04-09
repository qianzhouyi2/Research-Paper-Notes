---
title: "Indirect Local Attacks for Context-aware Semantic Segmentation Networks"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/03_Indirect_Local_Attacks_for_Context_aware_Semantic_Segmentation_Network/1911.13038.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/1911.13038
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-1 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# Indirect Local Attacks for Context-aware Semantic Segmentation Networks

## Reading Status

- Batch: 1 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2019
- Venue: ECCV
- Verification source: arxiv
- Verified title: Indirect Local Attacks for Context-aware Semantic Segmentation Networks

## Problem and Method (From Extracted Evidence)

- Evidence summary: Recently, deep networks have achieved impressive semantic segmentation performance, in particular thanks to their use of larger contextual information. In this paper, we show that the resulting networks are sensitive not only to global attacks, where perturbations affect the entire input image, but also to indirect local attacks where perturbations are confined to a small image region that does not overlap with the area that we aim to fool.
- Method hint: To this end, we introduce several indirect attack strategies, including adaptive local attacks, aiming to find the best image location to perturb, and universal local attacks. Furthermore, we propose attack detection techniques both for the global image level and to obtain a pixel-wise localization of the fooled regions.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Indirect Local Segmentation Attacks
- Indirect Local Attacks
- Adaptive Attacks
- Universal Local Attacks
- Adversarial Attack Detection
- Conclusion

## Extracted Themes

- Theme tags pending deeper full-text parsing. ^[ambiguous]

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Krishna Kanth Nakka]]
- [[entities/Mathieu Salzmann]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/1911.13038
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

