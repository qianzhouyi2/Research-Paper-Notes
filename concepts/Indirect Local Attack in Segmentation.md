---
title: Indirect Local Attack in Segmentation
category: concept
tags:
  - concept
  - adversarial-attack
  - semantic-segmentation
  - robustness
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh.md
  - https://arxiv.org/abs/1911.13038
  - https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500596.pdf
created: 2026-04-09
updated: 2026-04-10
summary: 间接局部攻击指只扰动一小块且不与目标区域重叠的区域，却能借助分割模型的长程上下文依赖误导远处像素预测。
provenance:
  extracted: 0.78
  inferred: 0.20
  ambiguous: 0.02
---

# Indirect Local Attack in Segmentation

## 定义

在语义分割里，攻击者不一定要直接修改想要骗过的那块区域；只要扰动一个不重叠的小局部区域，也可能通过模型的上下文建模间接带坏远处像素的预测。

## 为什么会成立

- 现代分割模型会利用较大的感受野或显式上下文聚合来决定每个像素类别。
- 这意味着局部扰动不只影响邻近像素，还可能沿着上下文依赖传播到更远区域。^[inferred]
- 因而“局部下手、远处出错”在 segmentation 里是可行的，不像很多分类攻击那样只关心全图单标签。

## 攻击形态

- 自适应局部攻击：寻找最值得扰动的位置。
- 通用局部攻击：学习可复用的小区域扰动模式。

## 启发

- 长程上下文既提升了分割精度，也可能扩大攻击影响范围。
- 这解释了为什么后续工作里，线状噪声或其他局部结构化扰动即使不覆盖整个目标区域，依然可能有效。

## 联网补充

- arXiv 1911.13038 明确提出：分割网络不仅会被全局扰动攻击，也会被“不与目标区域重叠”的间接局部扰动攻击。
- 该论文还指出，更依赖大上下文的高精度分割网络，往往对这类间接局部攻击更敏感。

## 关联

- [[concepts/Semantic Segmentation]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Discrete Linear Noise]]
- [[concepts/SegPGD]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: 在语义分割里，攻击者不一定要直接修改想要骗过的那块区域；只要扰动一个不重叠的小局部区域，也可能通过模型的上下文建模间接带坏远处像素的预测。
- Primary online sources used in this pass:
- https://arxiv.org/abs/1911.13038
- https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500596.pdf
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

