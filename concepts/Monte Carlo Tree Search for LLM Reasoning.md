---
title: Monte Carlo Tree Search for LLM Reasoning
category: concept
tags:
  - concept
  - llm
  - reasoning
  - mcts
sources:
  - notes/Monte Carlo Tree Search A Review of Recent Modifications and Applications.md
  - notes/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report.md
created: 2026-04-08
updated: 2026-04-08
summary: MCTS 在 LLM 推理中提供了可扩展搜索框架，可与自评估和自修正机制结合提升复杂任务求解率。
provenance:
  extracted: 0.83
  inferred: 0.15
  ambiguous: 0.02
---

# Monte Carlo Tree Search for LLM Reasoning

将采样式生成改造为“搜索-评估-回传”闭环，适合高难度推理任务。

## 联网补充

- MCTS 在 LLM 里真正增加的是“可回传的搜索控制”：选择、扩展、评估、回传把自由生成改造成可反复修正的决策过程。
- 它最适合部分解可以被打分的难题；若价值函数噪声太大，树策略会把错误启发式系统性放大。

## 关联页面

- [[references/Monte Carlo Tree Search A Review of Recent Modifications and Applications]]
- [[references/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report]]
- [[synthesis/Structured Reasoning Methods for LLMs]]


