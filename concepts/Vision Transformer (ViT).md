---
title: Vision Transformer (ViT)
category: concept
tags:
  - concept
  - architecture
  - pretraining
  - representation
  - robustness
sources:
  - https://arxiv.org/abs/2010.11929
  - https://proceedings.mlr.press/v139/touvron21a.html
  - https://arxiv.org/abs/2103.14030
  - https://arxiv.org/abs/2111.06377
  - https://arxiv.org/abs/2304.07193
  - https://arxiv.org/abs/2304.02643
  - https://arxiv.org/abs/2408.00714
created: 2026-04-09
updated: 2026-04-10
summary: ViT 将图像切分为 patch token 并使用 Transformer 编码器进行全局建模，已从分类 backbone 演进为视觉基础模型核心路线。
provenance:
  extracted: 0.81
  inferred: 0.16
  ambiguous: 0.03
---

# Vision Transformer (ViT)

## 定义

Vision Transformer（ViT）将图像切分为固定大小 patch（如 16x16），每个 patch 线性投影为 token，再送入 Transformer 编码器做全局建模。

## 核心机制

- Patchify + Linear Projection：把 2D 图像转成 token 序列。
- Position Embedding：补足空间位置信息。
- Self-Attention：直接建模远距离依赖，不依赖卷积感受野逐层扩张。
- 通常配合大规模预训练，再迁移到下游任务。^[inferred]

## 发展脉络（简）

- ViT：提出纯 Transformer 视觉分类路线。
- DeiT：降低训练门槛，使 ImageNet 规模下的训练更可行。
- Swin：引入分层结构与 shifted window，推进到检测/分割等密集任务。
- MAE：高掩码率自监督预训练，提高数据效率与迁移能力。
- DINOv2 / SAM / SAM 2：推动 ViT 路线向通用视觉表征与可提示分割扩展。^[inferred]

## 联网补充

- 2020-10（arXiv:2010.11929）：ViT 明确提出“图像可视作 patch 序列”，并展示大规模预训练下的强迁移能力。
- 2020-12（DeiT, PMLR 2021）：通过蒸馏与训练策略，使 data-efficient 的 ViT 训练在更常见算力条件下可落地。
- 2021-03（arXiv:2103.14030）：Swin 以 shifted windows 构造分层视觉 Transformer，成为密集预测主流 backbone 之一。
- 2021-11（arXiv:2111.06377）：MAE 采用高掩码率重建式预训练，推动 ViT 自监督规模化。
- 2023-04（arXiv:2304.07193 / 2304.02643）：DINOv2 与 SAM 分别代表“通用视觉特征”和“promptable segmentation”两条 ViT 基础模型化路径。
- 2024-08（arXiv:2408.00714）：SAM 2 将同一路线扩展到图像与视频统一分割。

## 关联

- [[concepts/Semantic Segmentation]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[concepts/SimBA (Simple Black-box Attack)]]
- [[concepts/Square Attack]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: Vision Transformer（ViT）将图像切分为固定大小 patch（如 16x16），每个 patch 线性投影为 token，再送入 Transformer 编码器做全局建模。
- Primary online sources used in this pass:
- https://arxiv.org/abs/2010.11929
- https://proceedings.mlr.press/v139/touvron21a.html
- https://arxiv.org/abs/2103.14030
- https://arxiv.org/abs/2111.06377
- https://arxiv.org/abs/2304.07193
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

