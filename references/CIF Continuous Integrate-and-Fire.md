---
title: CIF Continuous Integrate-and-Fire
category: reference
tags:
  - paper
  - asr
  - sequence-modeling
sources:
  - notes/CIF Continuous Integrate-and-Fire.md
created: 2026-04-08
updated: 2026-04-10
summary: CIF 通过连续积分放电机制统一单调软对齐、边界定位与端到端训练，是语音序列建模的重要对齐方法。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# CIF Continuous Integrate-and-Fire

## 摘要卡

- 原始笔记：[[notes/CIF Continuous Integrate-and-Fire]]
- 核心结论：CIF 用连续积分放电机制，把“单调软对齐、边界定位、端到端训练”三者有效统一到 ASR 中。
- 联网核验：arXiv 列作者为 Linhao Dong、Bo Xu，且 v4 comments 标注 “To appear at ICASSP 2020”。

## 细化方法锚点

- 对齐主机制：累计权重达到阈值触发发射，支持可微分训练。
- 数量控制：缩放策略与数量损失共同约束发射次数。
- 推理补偿：尾部处理减少末端信息丢失。

## 关联页面

- [[concepts/Continuous Integrate-and-Fire Alignment]]
- [[concepts/Quantity-Aware Emission Control in CIF]]
- [[entities/Linhao Dong]]
- [[entities/Bo Xu]]
- [[references/Hyena Hierarchy Towards Larger Convolutional Language Models]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]
