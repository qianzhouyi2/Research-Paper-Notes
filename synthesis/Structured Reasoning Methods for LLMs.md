---
title: Structured Reasoning Methods for LLMs
category: synthesis
tags:
  - synthesis
  - llm
  - reasoning
  - cot
sources:
  - notes/Tree of Thoughts Deliberate Problem Solving with Large Language Models.md
  - notes/Graph of Thoughts Solving Elaborate Problems with Large Language Models.md
  - notes/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models.md
  - notes/Synergy-of-Thoughts Eliciting Efficient Reasoning in Hybrid Language Models.md
  - notes/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations.md
created: 2026-04-08
updated: 2026-04-09
summary: 结构化推理方法正从线性 CoT 走向搜索、图结构与过程监督，核心趋势是以可控计算开销换取更稳定的复杂推理质量。
provenance:
  extracted: 0.72
  inferred: 0.25
  ambiguous: 0.03
---

# Structured Reasoning Methods for LLMs

## 共性趋势

- 从单链条推理扩展到树搜索和图结构推理。
- 从“生成答案”扩展到“评估与修正思维过程”。
- 从固定推理预算扩展到按难度动态分配推理计算。^[inferred]

## 代表页面

- [[references/Tree of Thoughts Deliberate Problem Solving with Large Language Models]]
- [[references/Graph of Thoughts Solving Elaborate Problems with Large Language Models]]
- [[references/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models]]
- [[references/Synergy-of-Thoughts Eliciting Efficient Reasoning in Hybrid Language Models]]
- [[references/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations]]

## 关联概念

- [[concepts/Tree of Thoughts Reasoning]]
- [[concepts/Graph of Thoughts Reasoning]]
- [[concepts/Monte Carlo Tree Search for LLM Reasoning]]
- [[concepts/Implicit Chain-of-Thought Internalization]]
- [[synthesis/Search-Based Deliberate Reasoning Landscape]]
- [[synthesis/Process Supervision and CoT Internalization]]

## 联网补充

- ToT/GoT/Beyond CoT 等工作一致表明：把推理过程外显为结构并允许搜索与合并，通常能比纯线性 CoT 在复杂任务上更稳定。
- Math-Shepherd 与 SoT 代表了另一趋势：将过程监督与推理时策略控制结合，形成“训练期 + 推理期”协同优化。
