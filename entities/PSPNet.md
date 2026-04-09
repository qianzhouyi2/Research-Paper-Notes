---
title: PSPNet
category: entity
tags:
- entity
- model
- semantic-segmentation
- robustness
sources:
- papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
- notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-09
updated: 2026-04-10
summary: PSPNet 是多尺度上下文聚合分割模型，在 DLA 中出现了最显著的 mIoU 下降案例之一。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# PSPNet

## 定义与定位

- PSPNet（Pyramid Scene Parsing Network）通过金字塔池化聚合多尺度上下文，是语义分割中的经典 CNN 架构。
- 在本库里，它是 DLA 评测里的核心对比对象之一，常被用来展示 query-efficient attack 的攻击强度。

## 在本批来源中的角色

- 作为 5 个 threat model 之一参与 Cityscapes 与 ADE20K 的统一评测。
- 与其他模型相比，PSPNet 在 Cityscapes 上给出了最醒目的“高起点 mIoU 被快速压低”结果。

## 观测

- Cityscapes 上，PSPNet 的 clean mIoU 为 77.83，DLA 在 50 queries 下可降至 2.14。
- 同一设置下，PSPNet 体现了“高 query 效率攻击也能快速打穿”的典型脆弱性。

## 关联

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Semantic Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[entities/FCN]]
- [[entities/DeepLabv3]]
- [[entities/SegFormer]]
- [[entities/MaskFormer]]

## Online Supplement (2026-04-10)

- This entity page is cross-checked online for attribution and role consistency across linked papers.
- Text anchor used: - PSPNet（Pyramid Scene Parsing Network）通过金字塔池化聚合多尺度上下文，是语义分割中的经典 CNN 架构。 - 在本库里，它是 DLA 评测里的核心对比对象之一，常被用来展示 query-efficient attack 的攻击强度。
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

