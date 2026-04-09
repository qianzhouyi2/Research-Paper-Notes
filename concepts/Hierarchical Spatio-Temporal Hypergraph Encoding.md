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
updated: 2026-04-09
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

## 联网补充

- GPT-ST 摘要把 hierarchical spatial pattern encoding networks 列为核心组件之一，用于捕获 customized representations 以及 intra-/inter-cluster semantic relationships。
- 这类层级结构最适合多尺度时空依赖明显的数据；若区域关系本身接近平坦图，层级编码的额外复杂度未必能兑现收益。

## 关联页面

- [[references/GPT-ST Spatio-Temporal Pretraining]]
- [[concepts/Cluster-Aware Masked Pretraining]]
- [[synthesis/Structured Spatio-Temporal Representation Learning]]



