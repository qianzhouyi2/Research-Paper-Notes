---
title: Search-Based Deliberate Reasoning Landscape
category: synthesis
tags:
  - synthesis
  - llm
  - reasoning
  - search
sources:
  - notes/Tree of Thoughts Deliberate Problem Solving with Large Language Models.md
  - notes/Graph of Thoughts Solving Elaborate Problems with Large Language Models.md
  - notes/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report.md
  - notes/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models.md
  - notes/Demystifying Chains, Trees, and Graphs of Thoughts.md
created: 2026-04-08
updated: 2026-04-10
summary: 该主题聚焦“显式搜索结构 + 过程评估反馈”的深度推理路径，比较树/图/MCTS 三类展开机制的计算与效果权衡。
provenance:
  extracted: 0.76
  inferred: 0.22
  ambiguous: 0.02
---

# Search-Based Deliberate Reasoning Landscape

## 综合结论

- 树搜索在分支控制上更直观，图搜索在状态复用与合并上更灵活。
- MCTS 与自评估结合可在固定预算下提高难题求解稳定性。^[inferred]

## 代表页面

- [[references/Tree of Thoughts Deliberate Problem Solving with Large Language Models]]
- [[references/Graph of Thoughts Solving Elaborate Problems with Large Language Models]]
- [[references/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report]]
- [[references/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models]]
- [[references/Demystifying Chains, Trees, and Graphs of Thoughts]]

## 关联概念

- [[concepts/Tree of Thoughts Reasoning]]
- [[concepts/Graph of Thoughts Reasoning]]
- [[concepts/Monte Carlo Tree Search for LLM Reasoning]]
- [[concepts/Monte Carlo Tree Self-Refine]]

## 联网补充

- ToT 把“深思”形式化为可搜索树，GoT 则扩展到任意图结构并支持节点合并/反馈回路，覆盖更复杂的思维拓扑。
- MCTS 综述中的选择-扩展-模拟-回传范式可直接映射到 LLM 推理搜索控制，适合作为可复用算法骨架。
