---
title: Process-Supervised Step Verification
category: concept
tags:
  - concept
  - llm
  - reasoning
  - verification
sources:
  - notes/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations.md
created: 2026-04-08
updated: 2026-04-10
summary: 对推理过程的中间步骤进行细粒度监督与反馈，提升复杂题目的可控性和正确率。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Process-Supervised Step Verification

区别于只看最终答案的监督方式，该机制直接评估中间推理步骤并回传奖励信号。

## 联网补充

- Math-Shepherd 把步骤好坏定义成“从这一步继续采样，是否还能高概率到达正确终答”，这比只看最终对错更适合长链数学推理。
- 这种验证很依赖终答案可校验、且能采样后续 continuation；离开可验证任务后，步骤监督会更难自动构造。

## 关联页面

- [[references/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations]]
- [[concepts/Implicit Chain-of-Thought Internalization]]
- [[synthesis/LLM Reasoning Search and Verification]]
