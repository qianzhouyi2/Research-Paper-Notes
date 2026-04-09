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
- papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving
  into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-08
updated: 2026-04-10
summary: 在 DLA 对比中，SegFormer 是相对更难攻击的分割模型之一，体现了较强的黑盒鲁棒性表现。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# SegFormer

## 联网补充

- SegFormer 论文把它定义为层次化 Transformer 编码器加轻量 MLP 解码器的语义分割框架，并明确强调无需位置编码。
- 在本库里，SegFormer 主要作为较难被黑盒决策攻击压垮的分割目标，用来承接 Transformer 分割模型的鲁棒性讨论。

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
- [[entities/FCN]]
- [[entities/PSPNet]]
- [[entities/DeepLabv3]]
- [[entities/MaskFormer]]

## Online Supplement (2026-04-10)

- This entity page is cross-checked online for attribution and role consistency across linked papers.
- Text anchor used: - SegFormer 论文把它定义为层次化 Transformer 编码器加轻量 MLP 解码器的语义分割框架，并明确强调无需位置编码。 - 在本库里，SegFormer 主要作为较难被黑盒决策攻击压垮的分割目标，用来承接 Transformer 分割模型的鲁棒性讨论。
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

