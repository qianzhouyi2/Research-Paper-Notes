---
title: DeepLabv3
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
updated: 2026-04-09
summary: DeepLabv3 以 atrous convolution 与 ASPP 建模多尺度上下文，在 DLA 评测中同样表现出显著脆弱性。
provenance:
  extracted: 0.85
  inferred: 0.13
  ambiguous: 0.02
---

# DeepLabv3

## 定义与定位

- DeepLabv3 通过 atrous convolution（空洞卷积）与 ASPP（Atrous Spatial Pyramid Pooling）提升多尺度语义建模能力。
- 在本库里，DeepLabv3 与 FCN、PSPNet 一起构成 CNN 分割模型对照组，用于对比 Transformer 分割模型的黑盒鲁棒性。

## 在本批来源中的角色

- 作为 DLA 主实验中被攻击的 5 个分割模型之一参与双数据集评测。
- 在固定查询预算下，DeepLabv3 的 mIoU 可被显著压低，体现 dense prediction 黑盒脆弱性。

## 观测

- Cityscapes 上，DLA 在 200 queries 下可把 DeepLabv3 的 mIoU 降到约 1.71。
- 对比 SegFormer，DeepLabv3 在相同预算下通常更容易被攻击到更低 mIoU。^[inferred]

## 关联

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Semantic Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[entities/FCN]]
- [[entities/PSPNet]]
- [[entities/SegFormer]]
- [[entities/MaskFormer]]

