---
title: Removal Smoothing for CoT Internalization
category: concept
tags:
  - concept
  - llm
  - cot
  - optimization
sources:
  - notes/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step.md
created: 2026-04-08
updated: 2026-04-10
summary: 在逐步移除 CoT token 的训练中引入随机平滑偏移，缓解目标分布突变导致的优化不稳定。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Removal Smoothing for CoT Internalization

该机制通过概率化移除策略让训练目标连续过渡，减少突然失配引发的性能崩塌。

## 联网补充

- Removal smoothing 会在每个阶段随机多移除一些 rationale token，把原本离散跳变的目标函数改造成更平滑的课程。
- 这不是锦上添花，而是稳定训练的关键件之一；没有它时，内化过程很容易在移除点附近突然崩掉。

## 关联页面

- [[references/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step]]
- [[concepts/Implicit Chain-of-Thought Internalization]]
- [[synthesis/Process Supervision and CoT Internalization]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Removal Smoothing for CoT Internalization category: concept tags: - concept - llm - cot - optimization sources: - notes/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step.md created: 2026-04-08 updated: 2026-04-08 summary: 在...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

