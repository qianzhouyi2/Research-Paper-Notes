---
title: "On the Robustness of Semantic Segmentation Models to Adversarial Attacks"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/08_On_the_Robustness_of_Semantic_Segmentation_Models_to_Adversarial_Attac/1711.09856.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/1711.09856
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-2 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# On the Robustness of Semantic Segmentation Models to Adversarial Attacks

## Reading Status

- Batch: 2 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2017
- Venue: TPAMI
- Verification source: arxiv
- Verified title: On the Robustness of Semantic Segmentation Models to Adversarial Attacks

## Problem and Method (From Extracted Evidence)

- Evidence summary: Deep Neural Networks (DNNs) have demonstrated exceptional performance on most recognition tasks such as image classification and segmentation. However, they have also been shown to be vulnerable to adversarial examples.
- Method hint: This phenomenon has recently attracted a lot of attention but it has not been extensively studied on multiple, large-scale datasets and structured prediction tasks such as semantic segmentation which often require more specialised networks with additional components such as CRFs, dilated convolutions, skip-connections and multiscale processing. In this paper, we present what to our knowledge is the first rigorous evaluation of adversarial attacks on modern semantic segmentation models, using two large-scale datasets.

## Local Paper Structure Evidence

- Structure extraction unavailable from local source format in this pass. ^[ambiguous]

## Extracted Themes

- Benchmark protocol design
- Attention refinement / regularization

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Anurag Arnab]]
- [[entities/Ondrej Miksik]]
- [[entities/Philip H. S. Torr]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/1711.09856
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

