---
title: Amortized Bayesian Inference for LLMs
category: concept
tags:
  - concept
  - llm
  - inference
  - bayesian
sources:
  - notes/Amortizing intractable inference in large language models.md
created: 2026-04-08
updated: 2026-04-10
summary: 将难解后验推理转化为可复用的近似策略学习，降低 LLM 推理时的采样与约束搜索成本。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# Amortized Bayesian Inference for LLMs

该范式通过离线学习“近似后验策略”，在在线推理阶段复用，避免每次都做昂贵的显式后验推断。

## 联网补充

- ICLR 2024 论文把 infilling、约束生成和 CoT 推理统一成后验采样问题，核心不是“找一个最好答案”，而是学习直接从目标后验分布采样。
- 这类摊销推断最适合目标密度或奖励可定义的任务；如果后验定义本身不稳，离线学到的采样器也会继承偏差。

## 关联页面

- [[references/Amortizing intractable inference in large language models]]
- [[concepts/Task Complexity-Aware Inference Budgeting]]
- [[synthesis/Probabilistic Inference-Time Control for LLMs]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: 该范式通过离线学习“近似后验策略”，在在线推理阶段复用，避免每次都做昂贵的显式后验推断。
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

