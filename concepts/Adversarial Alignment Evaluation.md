---
title: Adversarial Alignment Evaluation
category: concept
tags:
  - concept
  - llm-safety
  - adversarial-attack
  - evaluation
sources:
  - papers_sources/Are aligned neural networks adversarially aligned 2306.15447/main.tex
  - papers_sources/2306.15447.tar
created: 2026-04-08
updated: 2026-04-09
summary: 对抗性对齐评估强调先校准攻击器能力，再判断对齐防御强度，避免将“攻击失败”误判为“模型安全”。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# Adversarial Alignment Evaluation

## 定义

评估一个“已对齐”模型在最坏情况输入下是否仍保持对齐行为，而不是只看常规提示下的表现。

## 为什么重要

- 攻击失败可能来自攻击器不足，而非防御足够强。
- 若不先验证攻击器能力，安全评估会出现系统性乐观偏差。

## 关键实践

- 先构造“已知存在解”的任务，验证攻击器能否找到解。
- 再在真实目标上评估防御，分离“攻击能力”和“防御能力”两个变量。
- 在多模态场景需覆盖连续输入优化。^[inferred]

## 常见误区

- 只报告攻击成功率，不报告攻击器校准能力。
- 把离散文本攻击结论直接外推到多模态输入空间。^[inferred]

## 联网补充

- arXiv 2306.15447 明确指出，当前 NLP 攻击失败不能视为 text-only 对齐稳固的证据，因为即使优化攻击失败，暴力搜索仍可能找到对抗输入。
- 同文还展示多模态模型可通过图像扰动诱导任意不对齐行为，说明安全评测必须把连续输入通道纳入威胁模型。

## 关联页面

- [[references/Are aligned neural networks adversarially aligned]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Known-Solvable Attack Calibration]]
- [[concepts/Multimodal Adversarial Image Prompting]]
- [[synthesis/Adversarial Robustness Evaluation Patterns]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]


