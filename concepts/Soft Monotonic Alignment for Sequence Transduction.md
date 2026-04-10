---
title: Soft Monotonic Alignment for Sequence Transduction
category: concept
tags:
  - concept
  - alignment
  - sequence
  - asr
sources:
  - notes/CIF Continuous Integrate-and-Fire.md
created: 2026-04-08
updated: 2026-04-10
summary: 在保留单调对齐假设的前提下引入可微分软边界，提升序列转导中的对齐稳定性与可训练性。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Soft Monotonic Alignment for Sequence Transduction

该机制适用于语音到文本等顺序转导任务，在流式场景下兼顾对齐精度与推理延迟。

## 联网补充

- CIF 里的软单调对齐通过连续积分与阈值放电实现，边界触发时生成一个聚合后的 latent token 表示。
- 这让模型天然更适合流式或近单调转写任务；若源序列和目标序列需要大范围重排，这种归纳偏置就会变成限制。

## 关联页面

- [[references/CIF Continuous Integrate-and-Fire]]
- [[concepts/Continuous Integrate-and-Fire Alignment]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]
