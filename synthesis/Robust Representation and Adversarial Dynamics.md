---
title: Robust Representation and Adversarial Dynamics
category: synthesis
tags:
  - synthesis
  - robustness
  - adversarial
  - representation
sources:
  - notes/Are aligned neural networks adversarially aligned.md
  - notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
  - notes/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature.md
  - notes/Language Models Represent Space and Time.md
created: 2026-04-08
updated: 2026-04-09
summary: 该主题从攻击、评测与后处理防御三侧讨论“表示是否稳健”，强调任务语义与特征几何共同决定鲁棒性。
provenance:
  extracted: 0.74
  inferred: 0.24
  ambiguous: 0.02
---

# Robust Representation and Adversarial Dynamics

## 综合结论

- 攻击有效性往往来自表示空间中的脆弱方向，而不仅是输入扰动强度。
- 后处理稀疏化与过程化评测可部分修复脆弱表示，但仍受任务结构约束。^[inferred]

## 代表页面

- [[references/Are aligned neural networks adversarially aligned]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[references/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification]]
- [[references/Language Models Represent Space and Time]]

## 关联概念

- [[concepts/Adversarial Alignment Evaluation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/Mean-Centered Feature Sparsification]]
- [[concepts/Proximal L0 Feature Sparsification]]

## 联网补充

- Are-aligned 与 DLA 结果共同说明：表示稳健性必须在对抗优化与黑盒查询约束下评估，常规分布测试不足以覆盖风险。
- MeanSparse 证明后训练特征稀疏化可带来鲁棒增益，提示“表示几何后处理”是低成本防御的一条可行路线。
