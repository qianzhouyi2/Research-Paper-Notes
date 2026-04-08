---
title: SegFormer
category: entity
tags:
  - entity
  - model
  - semantic-segmentation
  - transformer
  - robustness
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-08
updated: 2026-04-08
summary: 在 DLA 对比中，SegFormer 是相对更难攻击的分割模型之一，体现了较强的黑盒鲁棒性表现。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# SegFormer

## 在本批来源中的角色

- 作为被攻击的语义分割模型之一参与主实验。
- 在多个攻击方法下，相比部分 CNN 模型表现出更高鲁棒性。

## 观测

- 在 Cityscapes 与 ADE20K 的对比结果中，DLA 仍能显著降低其 mIoU，但降幅相对部分模型更受限。
- 论文将其鲁棒性优势部分归因于 Transformer 结构。^[inferred]

## 关联

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]

