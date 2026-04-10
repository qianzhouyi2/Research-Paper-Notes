---
title: GFlowNet Posterior Sampling for Text Generation
category: concept
tags:
  - concept
  - llm
  - sampling
  - inference
sources:
  - notes/Amortizing intractable inference in large language models.md
created: 2026-04-08
updated: 2026-04-10
summary: 利用 GFlowNet 学习可归一化的生成流，实现对受约束文本后验分布的近似采样。
provenance:
  extracted: 0.79
  inferred: 0.19
  ambiguous: 0.02
---

# GFlowNet Posterior Sampling for Text Generation

相比仅优化单一路径概率，GFlowNet 更强调覆盖多样高质量解并保持采样分布可控。

## 联网补充

- GFlowNet 微调追求的是“按奖励比例采样整条文本分布”，而不是只把概率压到单个最优序列上，这正是它适合多峰后验的原因。
- 相比 PPO 式奖励最大化，它更强调多样性保留；代价是训练更依赖探索、回放与稳定的分布匹配目标。

## 关联页面

- [[references/Amortizing intractable inference in large language models]]
- [[concepts/Amortized Bayesian Inference for LLMs]]
- [[synthesis/Probabilistic Inference-Time Control for LLMs]]
