---
title: Semantic Segmentation Robustness Corpus 2019-2026
category: synthesis
tags:
  - synthesis
  - semantic-segmentation
  - adversarial-robustness
  - literature-map
sources:
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.md
created: 2026-04-09
updated: 2026-04-10
summary: 汇总 2019-2026 年语义分割鲁棒性 28 篇论文资源，形成攻击、训练、认证与评测四条主线。
provenance:
  extracted: 0.86
  inferred: 0.11
  ambiguous: 0.03
---

# Semantic Segmentation Robustness Corpus 2019-2026

## 概览

- 语料规模：28 篇（2019-2026）。
- 下载结果：28/28 成功。
- 资源类型：22 篇 arXiv TeX 源码（`.tar`），6 篇 PDF 回退。
- 本地入口：[[papers_sources/semantic_segmentation_robustness_20260409/download_report]]

## 主题分层

- 攻击方法：query-limited black-box、patch attack、context-aware local attack、LiDAR segmentation attack。
- 防御与训练：SegPGD-AT、divide-and-conquer adversarial training、attention refinement、painting-by-numbers、erosion attack training。
- 可认证与验证：reachability verification、randomized smoothing、patch-certified defense。
- 评测与基准：robustness benchmarking、reliability evaluation、SemSegBench/DetecBench。

## 可复用结论

- 语义分割鲁棒性评测不能只看单一白盒强攻，query budget 与任务特化指标同样关键。^[inferred]
- 训练侧改进（如 SegPGD-AT、dynamic adversarial training）与评测协议改进（如 reliability benchmark）需要同步推进。^[inferred]
- 认证方法已覆盖 segmentation，但在高分辨率与 patch 场景仍有性能-保真权衡。^[inferred]

## 不确定性标记

- 2025-2026 的 arXiv 预印本版本可能继续迭代，后续引用时建议固定到具体版本号或 commit。^[ambiguous]

## 关联

- [[concepts/Semantic Segmentation]]
- [[concepts/SegPGD]]
- [[concepts/Indirect Local Attack in Segmentation]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]
- [[synthesis/Adversarial Robustness Evaluation Patterns]]
- [[journal/Semantic Segmentation Robustness Corpus Update 2026-04-09]]
