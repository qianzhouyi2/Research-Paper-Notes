---
title: Mixture-of-Agents Aggregation
category: concept
tags:
  - concept
  - llm
  - multi-agent
  - orchestration
sources:
  - notes/Mixture-of-Agents Enhances Large Language Model Capabilities.md
  - notes/Chain of Agents Large Language Models Collaborating on Long-Context Tasks.md
created: 2026-04-08
updated: 2026-04-10
summary: 通过分层代理协作与聚合策略整合多模型候选解，提升长上下文与复杂任务的整体输出质量。
provenance:
  extracted: 0.81
  inferred: 0.17
  ambiguous: 0.02
---

# Mixture-of-Agents Aggregation

核心在于“分工求解 + 聚合裁决”，让不同代理在同一问题上形成互补。

## 联网补充

- MoA 中 aggregator 的角色不是简单投票，而是读取多份候选后做综合改写，因此 proposer 多样性本身就是性能来源的一部分。
- 论文也表明质量提升和成本几乎同步上升；层数、agent 数量、模型异构度都是需要显式调的预算变量。

## 关联页面

- [[references/Mixture-of-Agents Enhances Large Language Model Capabilities]]
- [[references/Chain of Agents Large Language Models Collaborating on Long-Context Tasks]]
- [[concepts/Multi-Agent LLM Orchestration]]
- [[synthesis/Multi-Agent LLM Collaboration Landscape]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Mixture-of-Agents Aggregation category: concept tags: - concept - llm - multi-agent - orchestration sources: - notes/Mixture-of-Agents Enhances Large Language Model Capabilities.md - notes/Chain of Agents Large Language Models Collaborating on Long...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

