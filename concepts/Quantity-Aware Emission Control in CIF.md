---
title: Quantity-Aware Emission Control in CIF
category: concept
tags:
  - concept
  - asr
  - alignment
  - sequence-modeling
sources:
  - papers_sources/Research-Paper-Notes/CIF.md
  - notes/CIF Continuous Integrate-and-Fire.md
created: 2026-04-08
updated: 2026-04-10
summary: CIF 的数量感知发射控制通过缩放、数量损失与尾部处理三策略约束发射数量，稳定单调对齐训练与推理。
provenance:
  extracted: 0.88
  inferred: 0.1
  ambiguous: 0.02
---

# Quantity-Aware Emission Control in CIF

## 定义

在 CIF 对齐中显式控制发射数量，使积分放电次数与目标长度更一致，降低长度偏移导致的识别误差。

## 核心组件

- 权重缩放策略
- 数量损失 `L_qua`
- 尾部补发射与 `<EOS>` 处理

## 联网补充

- CIF 的 arXiv 摘要明确指出，必须加入额外 support strategies 来缓解 CIF 独有问题；在实现上对应缩放、数量损失与尾部处理三件套。
- 数量控制不是附属细节，而是让 soft firing 与目标长度、CE 训练和在线解码兼容的关键，否则发射次数会系统性漂移。

## 关联页面

- [[references/CIF Continuous Integrate-and-Fire]]
- [[concepts/Continuous Integrate-and-Fire Alignment]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]
