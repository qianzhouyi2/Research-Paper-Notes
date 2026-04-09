---
title: "SegPGD: An Effective and Efficient Adversarial Attack for Segmentation"
category: reference
tags:
  - paper
  - semantic-segmentation
  - adversarial-robustness
  - robust-evaluation
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/16_SegPGD_An_Effective_and_Efficient_Adversarial_Attack_for_Segmentation/2207.12391.tar
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - https://arxiv.org/abs/2207.12391
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-3 per-paper reading card with verified metadata, local abstract evidence, and reusable robustness notes."
provenance:
  extracted: 0.8
  inferred: 0.15
  ambiguous: 0.05
---

# SegPGD: An Effective and Efficient Adversarial Attack for Segmentation

## Reading Status

- Batch: 3 / 4
- Per-paper review status: completed (metadata + abstract-level local reading + cross-source verification)

## Metadata

- Year: 2022
- Venue: ECCV
- Verification source: arxiv
- Verified title: SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness

## Problem and Method (From Extracted Evidence)

- Evidence summary: Deep neural network-based image classifications are vulnerable to adversarial perturbations. The image classifications can be easily fooled by adding artificial small and imperceptible perturbations to input images.
- Method hint: As one of the most effective defense strategies, adversarial training was proposed to address the vulnerability of classification models, where the adversarial examples are created and injected into training data during training. The attack and defense of classification models have been intensively studied in past years.

## Local Paper Structure Evidence

- Extraction source: tex-sections
- Section/subsection headings observed in local source:
- Introduction
- Related Work
- SegPGD for Evaluating and Boosting Segmentation
- SegPGD: An Effective and Efficient Segmentation Attack
- Convergence Analysis of SegPGD
- Segmentation Adversarial Training with SegPGD
- Experiment
- Experimental Setting

## Extracted Themes

- Adversarial training strategy
- Attention refinement / regularization
- SegPGD family attack/training

## Experimental Context

- Dataset mentions in extracted text: Not explicitly named in extracted abstract snippet. ^[ambiguous]
- Local artifact type: tex_source

## Linked Author Entities

- [[entities/Jindong Gu]]
- [[entities/Hengshuang Zhao]]
- [[entities/Volker Tresp]]
- [[entities/Philip Torr]]

## Linked Concepts and Synthesis

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Segmentation Adversarial Attack Methods 2019-2026]]
- [[concepts/Segmentation Robustness Benchmark Protocol]]

## Online Verification Notes

- Primary verification link: https://arxiv.org/abs/2207.12391
- Local artifacts are stored under `papers_sources/semantic_segmentation_robustness_20260409`.

