---
title: FCN
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
summary: FCN 是经典全卷积分割基线，在 DLA 评测中可在低查询预算下被显著降级。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# FCN

## 定义与定位

- FCN（Fully Convolutional Network）把分类 CNN 改造成端到端像素级预测框架，是现代语义分割的早期基线之一。
- 在本库里，FCN 主要作为 CNN 分割模型基线，用于和 PSPNet、DeepLabv3、SegFormer、MaskFormer 做黑盒鲁棒性横向对比。

## 在本批来源中的角色

- 作为 DLA 主实验中的 5 个 threat model 之一参与评测。
- 在 Cityscapes 与 ADE20K 上都呈现“可被高效决策型攻击显著压低 mIoU”的特征。

## 观测

- Cityscapes 上，DLA 在 200 queries 下可把 FCN 的 mIoU 打到约 3.07。
- 与 SegFormer 相比，FCN 在同类攻击预算下通常更容易被压低到更低 mIoU。^[inferred]

## 关联

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Semantic Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[entities/PSPNet]]
- [[entities/DeepLabv3]]
- [[entities/SegFormer]]
- [[entities/MaskFormer]]

## Online Supplement (2026-04-10)

- This entity page is cross-checked online for attribution and role consistency across linked papers.
- Text anchor used: - FCN（Fully Convolutional Network）把分类 CNN 改造成端到端像素级预测框架，是现代语义分割的早期基线之一。 - 在本库里，FCN 主要作为 CNN 分割模型基线，用于和 PSPNet、DeepLabv3、SegFormer、MaskFormer 做黑盒鲁棒性横向对比。
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

