---
title: Known-Solvable Attack Calibration
category: concept
tags:
  - concept
  - llm-safety
  - adversarial-attack
  - evaluation
sources:
  - papers_sources/Are aligned neural networks adversarially aligned 2306.15447/main.tex
  - notes/Are aligned neural networks adversarially aligned.md
created: 2026-04-08
updated: 2026-04-09
summary: Known-solvable calibration 先验证攻击器能否解出已知可解样本，再判断模型是否真正鲁棒。
provenance:
  extracted: 0.86
  inferred: 0.12
  ambiguous: 0.02
---

# Known-Solvable Attack Calibration

## 定义

先构造“保证存在触发解”的样本集合，再测试攻击器能否找到这些解，用于分离“攻击器弱”与“模型强”。

## 价值

- 避免把攻击失败误判为对齐成功。
- 把鲁棒评估拆成两阶段：攻击能力校准、真实目标评估。

## 联网补充

- 2306.15447 的一个核心方法论是先在已知存在解的设置上校准攻击器能力，再评估对齐防御；否则低攻击成功率没有解释力。
- 这意味着安全评测必须区分“防御真的强”和“攻击器根本没找到路”两件事，尤其是在离散文本攻击容易失灵时。

## 关联页面

- [[references/Are aligned neural networks adversarially aligned]]
- [[concepts/Adversarial Alignment Evaluation]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]



