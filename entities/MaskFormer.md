---
title: MaskFormer
category: entity
tags:
- entity
- model
- semantic-segmentation
- transformer
- robustness
sources:
- papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
- notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-09
updated: 2026-04-10
summary: MaskFormer 采用 mask classification 范式，在 DLA 对比中比 SegFormer 更容易被压低 mIoU。
provenance:
  extracted: 0.83
  inferred: 0.15
  ambiguous: 0.02
---

# MaskFormer

## 定义与定位

- MaskFormer 以 mask classification 统一语义/实例/全景分割建模，并通过 Transformer decoder 预测类别与掩码。
- 在本库里，它作为 Transformer 分割模型代表之一，用于和 SegFormer 以及 CNN 分割模型比较黑盒鲁棒性。

## 在本批来源中的角色

- 作为 DLA 主实验中的 5 个 threat model 之一参与 Cityscapes 与 ADE20K 评测。
- 结果显示其鲁棒性整体弱于 SegFormer，但仍明显强于若干低效攻击基线。^[inferred]

## 观测

- Cityscapes 上，DLA 在 200 queries 下可把 MaskFormer 的 mIoU 降到约 2.78。
- 论文讨论中指出，MaskFormer 的 backbone 含 CNN 结构，可能是其鲁棒性不及 SegFormer 的原因之一。^[inferred]

## 关联

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Semantic Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[entities/FCN]]
- [[entities/PSPNet]]
- [[entities/DeepLabv3]]
- [[entities/SegFormer]]

## ?????2026-04-10?

- ?????????????????????????????
- ?????? - MaskFormer 以 mask classification 统一语义/实例/全景分割建模，并通过 Transformer decoder 预测类别与掩码。 - 在本库里，它作为 Transformer 分割模型代表之一，用于和 SegFormer 以及 CNN 分割模型比较黑盒鲁棒性。
- ????????????
- ??????????? URL???????????????^[ambiguous]
- ????????????arXiv / DOI / ???????????????????
- ?????????????????????

