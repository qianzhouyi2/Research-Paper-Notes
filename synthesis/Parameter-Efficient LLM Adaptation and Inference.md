---
title: Parameter-Efficient LLM Adaptation and Inference
category: synthesis
tags:
  - synthesis
  - llm
  - efficiency
  - adaptation
sources:
  - notes/LoRA Low-Rank Adaptation of Large Language Models.md
  - notes/Amortizing intractable inference in large language models.md
  - notes/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity.md
created: 2026-04-08
updated: 2026-04-10
summary: 该主题整合参数高效适配与推理时动态计算分配两条路线，目标是在性能与成本之间建立可控折中。
provenance:
  extracted: 0.75
  inferred: 0.22
  ambiguous: 0.03
---

# Parameter-Efficient LLM Adaptation and Inference

## 综合结论

- 训练侧通过低秩适配降低可训练参数，推理侧通过难度感知策略动态分配计算预算。
- 两条路线可组合：前者减少微调成本，后者减少在线推理成本。^[inferred]

## 代表页面

- [[references/LoRA Low-Rank Adaptation of Large Language Models]]
- [[references/Amortizing intractable inference in large language models]]
- [[references/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity]]

## 关联概念

- [[concepts/Parameter-Efficient Fine-Tuning for LLMs]]
- [[concepts/Low-Rank Adaptation for LLMs]]
- [[concepts/Adaptive Compute Routing]]
- [[concepts/Task Complexity-Aware Inference Budgeting]]

## 联网补充

- LoRA 的核心结论是：在冻结预训练权重前提下，仅训练低秩适配矩阵即可完成有效迁移，显著降低显存与训练成本。
- 结合复杂度路由与摊销推理可形成“参数高效微调 + 推理时预算控制”的双层效率策略。

## Online Supplement (2026-04-10)

- This synthesis page is cross-checked online for cross-paper consistency and evaluation-scope alignment.
- Text anchor used: - 训练侧通过低秩适配降低可训练参数，推理侧通过难度感知策略动态分配计算预算。 - 两条路线可组合：前者减少微调成本，后者减少在线推理成本。^[inferred]
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

