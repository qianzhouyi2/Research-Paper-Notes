---
title: Delving into Decision-based Black-box Attacks on Semantic Segmentation
category: reference
tags:
  - paper
  - adversarial-attack
  - black-box-attack
  - semantic-segmentation
  - robustness
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh.md
  - papers_sources/2402.01220-source.tar
created: 2026-04-08
updated: 2026-04-08
summary: 首篇系统研究语义分割 decision-based 黑盒攻击的论文，提出 DLA 并在有限查询下显著降低多个模型 mIoU。
provenance:
  extracted: 0.88
  inferred: 0.10
  ambiguous: 0.02
---

# Delving into Decision-based Black-box Attacks on Semantic Segmentation

## 基本信息

- 年份：2024（arXiv 2402.01220）
- 任务：语义分割的 decision-based black-box attack
- 核心方法：DLA（Discrete Linear Attack）
- 原始笔记：[[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]

## 论文主张

- 语义分割中的 decision-based 攻击比图像分类更难，因为目标是像素级、多约束优化。
- 主要困难包含：
  - 优化目标不一致（缩小扰动可能导致 mIoU 回升）
  - 扰动交互（新扰动会破坏前一步攻击成果）
  - 参数空间复杂（查询预算下难以搜索）
- DLA 通过“代理指标 + 离散线性噪声 + 层次化校准”提高查询效率。

## 方法摘要

- 代理指标：以 mIoU 作为优化反馈。
- 扰动探索：使用离散线性噪声（horizontal / vertical / iterative）生成候选扰动。
- 扰动校准：根据指标变化，对局部区域执行符号翻转（flip）并保留改进。

## 细化方法锚点

- 两阶段查询分配：先探索方向性扰动，再做局部块级校准。
- 接受准则：仅当 `mIoU` 单调下降时更新，形成低开销搜索闭环。
- 搜索空间压缩：将连续像素扰动转为离散线性结构，降低高维搜索难度。

## 关键结果

- Cityscapes：PSPNet mIoU 可在 50 queries 内从 77.83 降到 2.14。
- ADE20K：整体也显著下降，但比 Cityscapes 更难攻击。
- 在 10 queries 极限预算下，DLA 仍优于对比方法。

## 局限与注意

- 论文主要是经验验证，理论解释相对有限。
- 对图中“视觉不可感知性”的结论依赖定性图示，缺少统一人类感知评测细则。^[ambiguous]

## 关联页面

- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Proxy Index mIoU Optimization]]
- [[concepts/Discrete Linear Noise]]
- [[concepts/Perturbation Interaction]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[synthesis/Decision-based Segmentation Attack Landscape]]
- [[synthesis/Adversarial Robustness Evaluation Patterns]]
- [[entities/Zhaoyu Chen]]
- [[entities/Wenqiang Zhang]]
- [[entities/Zhengyang Shan]]
- [[entities/SegFormer]]
- [[entities/Cityscapes Dataset]]
- [[entities/ADE20K Dataset]]
