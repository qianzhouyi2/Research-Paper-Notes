---
title: LLM Reasoning Search and Verification
category: synthesis
tags:
  - synthesis
  - llm
  - reasoning
  - verification
sources:
  - notes/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report.md
  - notes/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations.md
  - notes/Tree of Thoughts Deliberate Problem Solving with Large Language Models.md
  - notes/Graph of Thoughts Solving Elaborate Problems with Large Language Models.md
created: 2026-04-08
updated: 2026-04-08
summary: 该主题聚焦“搜索式思维展开 + 过程级验证反馈”的组合范式，强调在可控推理预算下提升复杂任务稳定性。
provenance:
  extracted: 0.77
  inferred: 0.21
  ambiguous: 0.02
---

# LLM Reasoning Search and Verification

## 综合结论

- 搜索结构（Tree/Graph/MCTS）负责扩展候选思路，过程验证负责筛选与回传。
- 两者组合比单纯长链 CoT 更稳健，尤其在高难数学与复杂规划任务中更明显。^[inferred]

## 代表页面

- [[references/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report]]
- [[references/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations]]
- [[references/Tree of Thoughts Deliberate Problem Solving with Large Language Models]]
- [[references/Graph of Thoughts Solving Elaborate Problems with Large Language Models]]

## 关联概念

- [[concepts/Monte Carlo Tree Self-Refine]]
- [[concepts/Monte Carlo Tree Search for LLM Reasoning]]
- [[concepts/Tree of Thoughts Reasoning]]
- [[concepts/Graph of Thoughts Reasoning]]
- [[concepts/Process-Supervised Step Verification]]

