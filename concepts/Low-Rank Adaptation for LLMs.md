---
title: Low-Rank Adaptation for LLMs
category: concept
tags:
  - concept
  - llm
  - finetuning
  - parameter-efficient
sources:
  - notes/LoRA Low-Rank Adaptation of Large Language Models.md
created: 2026-04-08
updated: 2026-04-08
summary: 低秩适配通过在冻结主干权重上注入小规模低秩增量，实现参数高效微调并降低部署成本。
provenance:
  extracted: 0.9
  inferred: 0.08
  ambiguous: 0.02
---

# Low-Rank Adaptation for LLMs

LoRA 的核心价值是把“全量微调”改写为“低秩增量学习”，减少显存与训练成本。

## 关联页面

- [[references/LoRA Low-Rank Adaptation of Large Language Models]]
- [[synthesis/LLM Inference Efficiency and Scaling]]

