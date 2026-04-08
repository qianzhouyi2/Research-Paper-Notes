---
title: Two-Stage Multimodal CoT Pipeline
category: concept
tags:
  - concept
  - multimodal
  - cot
  - reasoning
sources:
  - notes/Multimodal Chain-of-Thought Reasoning in Language Models.md
created: 2026-04-08
updated: 2026-04-08
summary: 将多模态推理拆分为“生成中间推理线索 + 条件化答案生成”两阶段，以缓解幻觉并提升可解释性。
provenance:
  extracted: 0.81
  inferred: 0.17
  ambiguous: 0.02
---

# Two-Stage Multimodal CoT Pipeline

先生成可审阅的跨模态推理线索，再据此生成最终答案，能降低端到端黑盒推理的不稳定性。

## 联网补充

- 两阶段 Multimodal-CoT 把“生成 rationale”和“根据 rationale 给答案”拆开，方便定位到底是证据抽取错了，还是答案决策错了。
- 它能缓解 explanation quality 和 answer quality 脱钩的问题，但前一阶段一旦产出高置信幻觉，后一阶段仍会被连带污染。

## 关联页面

- [[references/Multimodal Chain-of-Thought Reasoning in Language Models]]
- [[concepts/Multimodal Chain-of-Thought Reasoning]]
- [[synthesis/Multimodal Composition and Reasoning]]


