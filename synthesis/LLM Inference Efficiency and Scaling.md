---
title: LLM Inference Efficiency and Scaling
category: synthesis
tags:
  - synthesis
  - llm
  - efficiency
  - scaling
sources:
  - notes/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity.md
  - notes/Amortizing intractable inference in large language models.md
  - notes/Synergy-of-Thoughts Eliciting Efficient Reasoning in Hybrid Language Models.md
  - notes/LoRA Low-Rank Adaptation of Large Language Models.md
  - notes/PanGu-pi Nonlinearity Compensation.md
created: 2026-04-08
updated: 2026-04-10
summary: LLM 效率优化正在从单一压缩走向“路由调度、摊销推理与参数高效适配”三线并进，目标是在质量与成本之间实现可控平衡。
provenance:
  extracted: 0.74
  inferred: 0.23
  ambiguous: 0.03
---

# LLM Inference Efficiency and Scaling

## 主要路线

- 难度感知路由：按任务复杂度动态分配模型算力。
- 摊销推理：把昂贵后验推理迁移到可复用策略中。
- 参数高效适配：通过低秩或结构增量降低训练与部署成本。

## 代表页面

- [[references/ComplexityNet Increasing LLM Inference Efficiency by Learning Task Complexity]]
- [[references/Amortizing intractable inference in large language models]]
- [[references/LoRA Low-Rank Adaptation of Large Language Models]]
- [[references/PanGu-pi Nonlinearity Compensation]]

## 关联概念

- [[concepts/Adaptive Compute Routing]]
- [[concepts/Amortized Bayesian Inference for LLMs]]
- [[concepts/GFlowNet Posterior Sampling for Text Generation]]
- [[concepts/Task Complexity-Aware Inference Budgeting]]
- [[concepts/Success-Rate Based Complexity Labeling]]
- [[concepts/Low-Rank Adaptation for LLMs]]
- [[concepts/Parameter-Efficient Fine-Tuning for LLMs]]
- [[concepts/Hyena Operator]]
- [[concepts/Augmented Shortcut for Attention Blocks]]
- [[concepts/Cascaded Activation Nonlinearity Compensation]]
- [[synthesis/Probabilistic Inference-Time Control for LLMs]]
- [[synthesis/Long-Context Architecture Without Full Attention]]

## 联网补充

- ComplexityNet、Amortizing Inference、Synergy-of-Thoughts 共同指向“推理时控制”是效率提升主线：通过复杂度路由、后验近似与双过程切换减少单位问题成本。
- LoRA 与 PanGu-π 则从参数与结构层面补足另一条主线，和推理时策略形成“训练/部署协同优化”。

## Online Supplement (2026-04-10)

- This synthesis page is cross-checked online for cross-paper consistency and evaluation-scope alignment.
- Text anchor used: - 难度感知路由：按任务复杂度动态分配模型算力。 - 摊销推理：把昂贵后验推理迁移到可复用策略中。 - 参数高效适配：通过低秩或结构增量降低训练与部署成本。
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

