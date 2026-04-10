---
title: Implicit Chain-of-Thought Internalization
category: concept
tags:
  - concept
  - llm
  - reasoning
  - cot
sources:
  - notes/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step.md
created: 2026-04-08
updated: 2026-04-10
summary: 隐式 CoT 内化通过逐步移除显式推理 token，使模型在不依赖长推理文本的情况下保持推理能力。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# Implicit Chain-of-Thought Internalization

目标是在训练阶段利用显式推理监督，在推理阶段减少对显式长链文本的依赖。

## 联网补充

- 显式 CoT 在这里更像训练时脚手架：模型先依赖步骤监督学会推理，再逐步把这些步骤压缩进隐藏状态。
- 内化可以显著降推理时 token 成本，但并不保证完整保留显式推理能力；一步把 rationale 全去掉通常会明显掉点。

## 关联页面

- [[references/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step]]
- [[references/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations]]
- [[synthesis/Structured Reasoning Methods for LLMs]]
