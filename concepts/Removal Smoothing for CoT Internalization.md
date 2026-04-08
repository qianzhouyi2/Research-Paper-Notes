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
updated: 2026-04-08
summary: 在逐步移除 CoT token 的训练中引入随机平滑偏移，缓解目标分布突变导致的优化不稳定。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Removal Smoothing for CoT Internalization

该机制通过概率化移除策略让训练目标连续过渡，减少突然失配引发的性能崩塌。

## 关联页面

- [[references/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step]]
- [[concepts/Implicit Chain-of-Thought Internalization]]
- [[synthesis/Process Supervision and CoT Internalization]]

