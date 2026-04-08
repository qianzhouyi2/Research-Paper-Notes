---
title: Continuous Integrate-and-Fire Alignment
category: concept
tags:
  - concept
  - asr
  - alignment
  - sequence-modeling
sources:
  - notes/CIF Continuous Integrate-and-Fire.md
created: 2026-04-08
updated: 2026-04-08
summary: Continuous Integrate-and-Fire 以连续积分放电机制实现单调软对齐与边界触发，是端到端语音序列对齐的重要方法。
provenance:
  extracted: 0.89
  inferred: 0.09
  ambiguous: 0.02
---

# Continuous Integrate-and-Fire Alignment

该机制将序列对齐与边界生成耦合到同一可训练流程中。

## 关键组成

- 连续积分放电主机制（单调软对齐）。
- 数量约束策略（缩放与数量损失）。
- 尾部补偿与结束标记处理。

## 关联页面

- [[references/CIF Continuous Integrate-and-Fire]]
- [[concepts/Quantity-Aware Emission Control in CIF]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]
