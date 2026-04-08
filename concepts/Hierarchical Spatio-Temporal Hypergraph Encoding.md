---
title: Hierarchical Spatio-Temporal Hypergraph Encoding
category: concept
tags:
  - concept
  - spatio-temporal
  - hypergraph
  - graph-learning
sources:
  - papers_sources/Research-Paper-Notes/GPT-ST预训练时空框架.md
  - notes/GPT-ST Spatio-Temporal Pretraining.md
created: 2026-04-08
updated: 2026-04-08
summary: 分层时空超图编码通过簇内关系建模与簇间关系迁移联合提升复杂时空结构表达能力。
provenance:
  extracted: 0.85
  inferred: 0.13
  ambiguous: 0.02
---

# Hierarchical Spatio-Temporal Hypergraph Encoding

## 定义

先编码节点到簇的局部关系，再建模簇到簇的高层迁移关系，并把高层信息回传到节点表示。

## 适用场景

- 交通流、区域活动等天然具有群组结构的时空预测任务。
- 需要同时利用局部相似性与跨群体迁移关系的任务。^[inferred]

## 关联页面

- [[references/GPT-ST Spatio-Temporal Pretraining]]
- [[concepts/Cluster-Aware Masked Pretraining]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]

