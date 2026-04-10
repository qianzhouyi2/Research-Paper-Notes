---
title: Adversarial Robustness Evaluation Patterns
category: synthesis
tags:
  - synthesis
  - adversarial-robustness
  - evaluation
  - methodology
sources:
  - papers_sources/Are aligned neural networks adversarially aligned 2306.15447/main.tex
  - papers_sources/MeanSparse 2406.05927/main.tex
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/main.tex
created: 2026-04-08
updated: 2026-04-10
summary: 综合三篇论文可得到鲁棒评估的三条主线：先校准攻击器能力、再做任务特定黑盒评测、最后用后处理策略做低成本鲁棒增益验证。
provenance:
  extracted: 0.73
  inferred: 0.24
  ambiguous: 0.03
---

# Adversarial Robustness Evaluation Patterns

## 核心观察

- [[references/Are aligned neural networks adversarially aligned]] 强调“攻击失败不等于模型安全”。
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]] 给出任务特定、查询受限下的黑盒评测路径。
- [[references/MeanSparse Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification]] 展示“后处理增强”可作为低成本鲁棒增益方案。

## 可复用评测框架

1. 先做攻击器能力校准（known-solvable / sanity check）。
2. 再做任务特化攻击评测（如分割任务下的代理指标与查询预算）。
3. 最后做防御侧增益验证（重训方案 + 后处理方案并行比较）。^[inferred]

## 对当前 wiki 的启发

- 评测页应区分“攻击器不足”与“防御有效”两个结论层级。
- 新增防御方法时，优先记录可迁移到现有模型的后处理路径。^[inferred]

## 关联页面

- [[concepts/Adversarial Alignment Evaluation]]
- [[concepts/Known-Solvable Attack Calibration]]
- [[concepts/Multimodal Adversarial Image Prompting]]
- [[concepts/Mean-Centered Feature Sparsification]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[synthesis/Alignment Robustness Evaluation Ladder]]
- [[synthesis/Robust Representation and Adversarial Dynamics]]

## 联网补充

- 《Are aligned neural networks adversarially aligned?》显示：模型在常规对齐测试中表现良好，并不等价于在对抗输入下依然安全，鲁棒评估需要显式 worst-case 视角。
- 结合 DLA（分割决策黑盒攻击）与 MeanSparse（后训练鲁棒增强）可形成“攻击能力校准 -> 任务攻击压测 -> 低成本防御增益验证”的闭环。
