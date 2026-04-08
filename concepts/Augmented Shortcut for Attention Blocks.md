---
title: Augmented Shortcut for Attention Blocks
category: concept
tags:
  - concept
  - llm
  - architecture
  - efficiency
sources:
  - papers_sources/Research-Paper-Notes/PanGu-π.md
  - notes/PanGu-pi Nonlinearity Compensation.md
created: 2026-04-08
updated: 2026-04-08
summary: 增强捷径在注意力块并行保留多样化特征路径，可缓解深层特征坍塌与秩收缩问题。
provenance:
  extracted: 0.84
  inferred: 0.14
  ambiguous: 0.02
---

# Augmented Shortcut for Attention Blocks

## 定义

在 MSA 主分支外引入结构化捷径分支，保持信息通路多样性并减轻重复线性变换导致的表示退化。

## 工程意义

- 可在较小增量参数下提升深层特征可分性。
- 常与非线性补偿激活联合使用。^[inferred]

## 关联页面

- [[references/PanGu-pi Nonlinearity Compensation]]
- [[concepts/LLM Nonlinearity Compensation]]
- [[synthesis/Long-Context Architecture Without Full Attention]]

