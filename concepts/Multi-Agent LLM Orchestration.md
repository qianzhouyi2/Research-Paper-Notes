---
title: Multi-Agent LLM Orchestration
category: concept
tags:
  - concept
  - llm
  - multi-agent
  - orchestration
sources:
  - notes/Chain of Agents Large Language Models Collaborating on Long-Context Tasks.md
  - notes/Mixture-of-Agents Enhances Large Language Model Capabilities.md
created: 2026-04-08
updated: 2026-04-10
summary: 多智能体编排通过角色分工、消息传递与聚合策略提升复杂任务表现，但同时引入额外成本与系统复杂度。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# Multi-Agent LLM Orchestration

## 关键组件

- 角色划分（worker / manager）
- 中间结果协议
- 最终聚合策略

## 联网补充

- CoA 和 MoA 对应了两种不同编排逻辑：前者强调链式通信压缩长上下文，后者强调 proposer-aggregator 分层综合候选答案。
- 因此多智能体收益主要来自协议设计而不是 agent 数量本身；消息接口和角色分工设计差，额外 agent 只会堆成本。

## 关联页面

- [[references/Chain of Agents Large Language Models Collaborating on Long-Context Tasks]]
- [[references/Mixture-of-Agents Enhances Large Language Model Capabilities]]
- [[synthesis/Multi-Agent LLM Collaboration Landscape]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Multi-Agent LLM Orchestration category: concept tags: - concept - llm - multi-agent - orchestration sources: - notes/Chain of Agents Large Language Models Collaborating on Long-Context Tasks.md - notes/Mixture-of-Agents Enhances Large Language Mode...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

