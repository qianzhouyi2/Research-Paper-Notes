---
title: LLaMA-Adapter
category: entity
tags:
- entity
- model
- multimodal
- llm-safety
sources:
- papers_sources/Are aligned neural networks adversarially aligned 2306.15447/main.tex
- notes/Are aligned neural networks adversarially aligned.md
created: 2026-04-08
updated: 2026-04-10
summary: LLaMA-Adapter 在多模态对抗评估中用于验证视觉扰动诱导失配输出的可行性。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# LLaMA-Adapter

## 联网补充

- LLaMA-Adapter 最早强调用 Zero-init Attention 和约 1.2M 可学习参数高效把 LLaMA 变成指令模型，后续 V2 又扩展到视觉指令场景。
- 在本库里，这个实体主要作为多模态对齐攻击对象出现，价值在于提供一种比全量微调更轻的适配路径对照。

## 关联论文

- [[references/Are aligned neural networks adversarially aligned]]

## 关联主题

- [[concepts/Multimodal Adversarial Image Prompting]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]
