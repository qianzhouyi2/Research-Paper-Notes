---
title: Alignment Robustness Evaluation Ladder
category: synthesis
tags:
  - synthesis
  - llm-safety
  - adversarial-attack
  - evaluation
sources:
  - papers_sources/Are aligned neural networks adversarially aligned 2306.15447/main.tex
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/main.tex
  - papers_sources/MeanSparse 2406.05927/main.tex
created: 2026-04-08
updated: 2026-04-09
summary: 对齐鲁棒评估可按“攻击器校准->任务化攻击->防御增益验证”三级阶梯执行，减少安全结论的乐观偏差。
provenance:
  extracted: 0.72
  inferred: 0.25
  ambiguous: 0.03
---

# Alignment Robustness Evaluation Ladder

## 三级阶梯

1. 攻击器校准：先验证攻击器能否解出 known-solvable 样本。
2. 任务化攻击：在具体任务约束下报告查询预算与效果退化曲线。
3. 防御验证：把训练内防御与后处理防御放到统一口径下对比。

## 对应论文

- [[references/Are aligned neural networks adversarially aligned]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[references/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification]]

## 关联概念

- [[concepts/Known-Solvable Attack Calibration]]
- [[concepts/Multimodal Adversarial Image Prompting]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/Mean-Centered Feature Sparsification]]

## 联网补充

- adversarial alignment 研究强调：对齐结论应在攻击者可优化输入的条件下复核，而不是只看平均场景表现。
- 语义分割场景的 decision-based 攻击与后训练鲁棒增强结果共同支持“分层评估梯度”：先校准攻击器，再做任务化压测，最后验证防御增益。
