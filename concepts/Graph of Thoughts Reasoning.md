---
title: Graph of Thoughts Reasoning
category: concept
tags:
  - concept
  - llm
  - reasoning
  - graph
sources:
  - notes/Graph of Thoughts Solving Elaborate Problems with Large Language Models.md
  - notes/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models.md
created: 2026-04-08
updated: 2026-04-08
summary: Graph of Thoughts 把推理过程表示为可组合图结构，通过节点变换与聚合提升复杂任务的推理质量与可控性。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# Graph of Thoughts Reasoning

将“推理步骤”从线性链扩展为图结构，使分支合并与重用更自然。

## 联网补充

- Graph-of-Thoughts 把 thought 组织成任意图，并显式支持生成、精炼、聚合三类变换，因此能表达跨分支合并而不只是树状回溯。
- 报告中的收益很大一部分来自跨路径重组；如果任务本身是单路径可解，图结构的额外控制成本未必划算。

## 关联页面

- [[references/Graph of Thoughts Solving Elaborate Problems with Large Language Models]]
- [[references/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models]]
- [[synthesis/Structured Reasoning Methods for LLMs]]


