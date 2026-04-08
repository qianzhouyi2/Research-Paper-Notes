---
title: Are aligned neural networks adversarially aligned?
category: reference
tags:
  - paper
  - llm-safety
  - adversarial-attack
  - multimodal
sources:
  - papers_sources/Are aligned neural networks adversarially aligned 2306.15447/main.tex
  - papers_sources/2306.15447.tar
created: 2026-04-08
updated: 2026-04-08
summary: 该论文提出“对抗性对齐”评估框架，指出文本攻击器能力不足会导致安全误判，并展示多模态模型可被对抗图像稳定诱导输出有害内容。
provenance:
  extracted: 0.87
  inferred: 0.11
  ambiguous: 0.02
---

# Are aligned neural networks adversarially aligned?

## 基本信息

- 年份：2023（NeurIPS）
- 任务：评估“对齐模型在最坏输入下是否仍对齐”
- 论文笔记：[[notes/Are aligned neural networks adversarially aligned]]

## 核心主张

- 现有 NLP 优化攻击经常失败，但这不能直接证明文本模型已具备对抗鲁棒性。
- 在“保证存在解”的构造测试中，攻击器仍会失败，说明评估能力本身是瓶颈。
- 多模态模型可通过图像连续扰动被稳定诱导到不对齐输出。

## 可复用方法

- 把“防御是否强”与“攻击器是否足够强”拆开验证。
- 在正式评估前先做可解性校准（known-solvable sanity check）。
- 多模态安全评估需要覆盖连续输入空间，而不只看离散文本提示。^[inferred]

## 细化方法锚点

- 文本攻击侧：以 harmful prefix 作为优化代理，降低长序列目标优化难度。
- 评测协议侧：先做 known-solvable 校准，再进入真实安全目标评估。
- 多模态侧：通过对抗图像 + teacher-forcing 目标序列测试跨模态失配风险。

## 边界与注意

- 文本端最终结论仍是“开放问题”，因为当前攻击器能力不足。^[ambiguous]
- 论文提出“更强 NLP 攻击可能复现多模态同级别脆弱性”，这一点仍待后续实证。^[ambiguous]

## 关联页面

- [[concepts/Adversarial Alignment Evaluation]]
- [[concepts/Known-Solvable Attack Calibration]]
- [[concepts/Multimodal Adversarial Image Prompting]]
- [[entities/Vicuna]]
- [[entities/OpenAssistant]]
- [[entities/Nicholas Carlini]]
- [[entities/Florian Tramer]]
- [[entities/Ludwig Schmidt]]
- [[entities/MiniGPT-4]]
- [[entities/LLaVA]]
- [[entities/LLaMA-Adapter]]
- [[synthesis/Adversarial Robustness Evaluation Patterns]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
