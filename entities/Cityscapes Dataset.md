---
title: Cityscapes Dataset
category: entity
tags:
- entity
- dataset
- semantic-segmentation
- adversarial-attack
sources:
- papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/main.tex
- notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-08
updated: 2026-04-10
summary: Cityscapes 是语义分割黑盒攻击评测的重要基准，DLA 在该数据集上表现出高查询效率攻击能力。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# Cityscapes Dataset

## 联网补充

- Cityscapes 官方页面把它定义为城市场景语义理解数据集，包含 5000 张高质量精标注图像、20000 张粗标注图像，覆盖 50 个城市。
- 在本库里，Cityscapes 是语义分割黑盒攻击的主评测场，重点不是类别本身，而是查询预算下 mIoU 能被压低多少。

## 关联论文

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]

## 关联主题

- [[concepts/Proxy Index mIoU Optimization]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]

## Online Supplement (2026-04-10)

- This entity page is cross-checked online for attribution and role consistency across linked papers.
- Text anchor used: - Cityscapes 官方页面把它定义为城市场景语义理解数据集，包含 5000 张高质量精标注图像、20000 张粗标注图像，覆盖 50 个城市。 - 在本库里，Cityscapes 是语义分割黑盒攻击的主评测场，重点不是类别本身，而是查询预算下 mIoU 能被压低多少。
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

