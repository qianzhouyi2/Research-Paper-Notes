---
title: Multimodal Adversarial Image Prompting
category: concept
tags:
  - concept
  - multimodal
  - adversarial-attack
  - llm-safety
sources:
  - papers_sources/Are aligned neural networks adversarially aligned 2306.15447/main.tex
  - notes/Are aligned neural networks adversarially aligned.md
created: 2026-04-08
updated: 2026-04-08
summary: 多模态对抗图像提示通过连续像素优化诱导对齐模型输出有害文本，是离散文本攻击之外的重要威胁面。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Multimodal Adversarial Image Prompting

## 定义

在视觉输入上执行梯度优化（如 PGD），以固定目标文本作为 teacher-forcing 监督，最大化目标有害输出出现概率。

## 适用边界

- 适用于可微视觉编码链路。
- 对纯文本模型不可直接使用，需要其他离散优化策略。^[inferred]

## 关联页面

- [[references/Are aligned neural networks adversarially aligned]]
- [[entities/MiniGPT-4]]
- [[entities/LLaVA]]
- [[entities/LLaMA-Adapter]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]

