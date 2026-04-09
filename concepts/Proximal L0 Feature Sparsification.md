---
title: Proximal L0 Feature Sparsification
category: concept
tags:
  - concept
  - robustness
  - optimization
  - sparsification
sources:
  - notes/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature.md
created: 2026-04-08
updated: 2026-04-10
summary: 以近端算子视角实现 L0 特征稀疏化，为后处理鲁棒增强提供可解释优化基础。
provenance:
  extracted: 0.83
  inferred: 0.15
  ambiguous: 0.02
---

# Proximal L0 Feature Sparsification

将均值中心化截断操作映射为近端优化近似，使后处理防御具备明确数学动机。

## 联网补充

- MeanSparse 把后处理稀疏化和 ℓ0 近端算子联系起来，使“截断小波动”从经验技巧变成有优化解释的操作。
- 它实际截断的是均值附近的通道内微小波动，而不是追求全局极致稀疏；这也是它能在不重训时维持 clean accuracy 的原因。

## 关联页面

- [[references/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification]]
- [[concepts/Mean-Centered Feature Sparsification]]
- [[synthesis/Adversarial Robustness Evaluation Patterns]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Proximal L0 Feature Sparsification category: concept tags: - concept - robustness - optimization - sparsification sources: - notes/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature.md created: 2026-04-08 updated: 2026-04...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

