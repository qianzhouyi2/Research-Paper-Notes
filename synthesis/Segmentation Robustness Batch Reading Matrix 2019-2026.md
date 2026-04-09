---
title: "Segmentation Robustness Batch Reading Matrix 2019-2026"
category: synthesis
tags:
  - synthesis
  - semantic-segmentation
  - adversarial-robustness
  - reading-matrix
sources:
  - workspace/seg_robustness_verified_2026-04-10.json
  - workspace/seg_robustness_local_reading_2026-04-10.json
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
created: 2026-04-10
updated: 2026-04-10
summary: "Batch-level matrix that maps all 28 papers to attack, training, certification, and reliability tracks."
provenance:
  extracted: 0.83
  inferred: 0.14
  ambiguous: 0.03
---

# Segmentation Robustness Batch Reading Matrix 2019-2026

## Batch 1 (Paper 1-7): Early Attack and Robustness Baselines

- Core track: redundant teacher-student robustness, systematic attack generation, indirect local attacks, baseline robustness benchmarking, and depth-assisted robustness.
- Main output: early-stage robustness framing moved from image-classification assumptions to segmentation-specific threat models.
- Representative links:
- [[references/On the Robustness of Redundant Teacher-Student Frameworks for Semantic Segmentation]]
- [[references/The Attack Generator - A Systematic Approach Towards Constructing Adversarial Attacks]]
- [[references/Indirect Local Attacks for Context-aware Semantic Segmentation Networks]]
- [[references/Benchmarking the Robustness of Semantic Segmentation Models (CVPR 2020)]]
- [[references/Improved Noise and Attack Robustness of Semantic Segmentation Models with Self-Supervised Depth Estimation]]

## Batch 2 (Paper 8-14): Verification and Cross-domain Expansion

- Core track: segmentation attack vulnerability profiling, relaxed-reachability verification, smoothing-based certification, divide-and-conquer robust training, LiDAR and medical settings, and physical patch evaluation.
- Main output: robustness moved from pure attack reports to formal guarantees and domain-specific risk analysis.
- Representative links:
- [[references/On the Robustness of Semantic Segmentation Models to Adversarial Attacks]]
- [[references/Robustness Verification of Semantic Segmentation Neural Networks using Relaxed Reachability]]
- [[references/Scalable Certified Segmentation via Randomized Smoothing]]
- [[references/Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation]]
- [[references/Towards Robust General Medical Image Segmentation]]

## Batch 3 (Paper 15-21): Stronger Attacks and Reliability Framing

- Core track: semantically stealthy attacks, SegPGD attack/training coupling, LiDAR demo attacks, certified patch defense, ensemble black-box transfer, proximal splitting attacks, and reliability auditing.
- Main output: evaluation shifted toward stronger attack baselines and explicit reliability narratives.
- Representative links:
- [[references/Towards Semantically Stealthy Adversarial Attacks Against Segmentation Models]]
- [[references/SegPGD - An Effective and Efficient Adversarial Attack for Segmentation]]
- [[references/Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation]]
- [[references/Ensemble-Based Blackbox Attacks on Dense Prediction]]
- [[references/Reliability in Semantic Segmentation - Are We on the Right Track]]

## Batch 4 (Paper 22-28): Protocol-centered Evaluation and New Benchmarks

- Core track: attention refinement against patch attacks, uncertainty-based detection, large-scale robustness evaluation protocols (SEA/PIR-AT), and benchmark suites (SemSegBench/DetecBench), plus RP-PGD and Erosion-Attack training.
- Main output: the field trend shifted from isolated method claims to protocol reliability and benchmark standardization.
- Representative links:
- [[references/Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement]]
- [[references/Uncertainty-Based Detection of Adversarial Attacks in Semantic Segmentation]]
- [[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]]
- [[references/SemSegBench and DetecBench - Benchmarking Reliability and Generalization Beyond Classification]]
- [[references/Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness (arXiv preprint)]]

## Cross-batch Conclusions

- Segmentation robustness research evolved from attack feasibility to standardized reliability reporting.
- Attack strength, training strategy, certification guarantees, and benchmark protocol must be reported together to avoid misleading robustness claims. ^[inferred]
- The 2025-2026 papers should be tracked with version-aware reading notes because preprint iteration risk is higher. ^[ambiguous]

## Links

- [[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]
- [[synthesis/Reliability and Benchmarking for Robust Segmentation]]
- [[journal/Segmentation Robustness Batch Ingest Plan 2026-04-10]]

## Online Supplement (2026-04-10)

- This synthesis page is cross-checked online for cross-paper consistency and evaluation-scope alignment.
- Text anchor used: - Core track: redundant teacher-student robustness, systematic attack generation, indirect local attacks, baseline robustness benchmarking, and depth-assisted robustness. - Main output: early-stage robustness framing moved from image-classification assumptions...
- Primary online sources used in this pass:
- https://doi.org/10.1109/cvprw.2019.00178
- https://arxiv.org/abs/1906.07077
- https://arxiv.org/abs/1911.13038
- https://arxiv.org/abs/1908.05005
- https://doi.org/10.1109/cvprw50498.2020.00168
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

