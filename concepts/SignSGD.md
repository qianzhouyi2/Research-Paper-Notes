---
title: SignSGD
category: concept
tags:
  - concept
  - optimization
  - efficiency
  - query-efficient
  - robustness
sources:
  - https://arxiv.org/abs/1810.05291
created: 2026-04-09
updated: 2026-04-10
summary: SignSGD 仅使用梯度符号更新参数，可显著压缩通信开销，并通过 majority vote 形成分布式鲁棒聚合。
provenance:
  extracted: 0.50
  inferred: 0.50
  ambiguous: 0.00
---

# SignSGD

## 定义

SignSGD 是只使用梯度符号（而非幅值）更新参数的优化方法：^[extracted]

\[
w_{t+1}=w_t-\eta \cdot sign(g_t), \quad g_t=\nabla_w L(w_t)
\]

## 直观理解

- 只保留“方向信息”（+/-），忽略精确梯度大小。^[extracted]
- 在分布式场景中可把通信从 32-bit/16-bit 梯度压缩到 1-bit 符号。^[inferred]
- 常见实现会配合 momentum 以提升稳定性。^[inferred]

## 联网补充

- 2018-10（arXiv:1810.05291）：SignSGD with Majority Vote 给出非凸情形收敛分析，并强调极低通信成本。^[extracted]
- 同一工作指出 majority vote 聚合对部分 Byzantine worker 具有鲁棒性，这也是其在分布式训练中的关键价值。^[extracted]

## 在黑盒攻击语境中的角色

- 黑盒攻击常受查询预算限制，sign-based 更新可作为轻量迭代策略之一。^[inferred]
- 但 SignSGD 本身是优化算法，不等同于具体攻击方法；真正攻击效果仍取决于目标函数与查询接口。^[inferred]

## 关联

- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/SimBA (Simple Black-box Attack)]]
- [[concepts/Square Attack]]
