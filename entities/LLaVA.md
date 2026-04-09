---
title: LLaVA
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
updated: 2026-04-09
summary: LLaVA 在多模态对齐攻击实验中被用于验证视觉对抗扰动对有害文本输出的诱导风险。
provenance:
  extracted: 0.83
  inferred: 0.15
  ambiguous: 0.02
---

# LLaVA

## 联网补充

- LLaVA 项目把它定义为连接视觉编码器与 Vicuna 的端到端多模态模型，并采用两阶段指令调优来获得通用视觉语言理解能力。
- 在本库里，LLaVA 的角色偏向被攻击的对齐模型，因此关注点落在鲁棒性与越狱面，而不是它本身的 benchmark 冲榜。

## 关联论文

- [[references/Are aligned neural networks adversarially aligned]]

## 关联主题

- [[concepts/Multimodal Adversarial Image Prompting]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]

