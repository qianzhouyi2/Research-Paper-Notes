---
title: Decision-based Segmentation Attack Landscape
category: synthesis
tags:
  - synthesis
  - adversarial-attack
  - semantic-segmentation
  - black-box-attack
  - robustness
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
created: 2026-04-08
updated: 2026-04-10
summary: DLA 展示了语义分割 decision-based 攻击可在极低查询下强力破坏 mIoU，提示鲁棒性评估应转向查询效率视角。
provenance:
  extracted: 0.79
  inferred: 0.19
  ambiguous: 0.02
---

# Decision-based Segmentation Attack Landscape

## 主题结论

- 语义分割黑盒攻击难点与图像分类不同，不能直接复用边界攻击直觉。
- 以查询效率为核心的攻击框架更能反映真实部署风险。
- DLA 路线强调三件事：指标稳定、搜索空间压缩、结构化局部校准。

## 与已有笔记的连接

- 在 [[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]] 中给出了完整方法和数据表。
- 与 [[synthesis/Alignment Robustness Evaluation Ladder]] 一起可形成“评测协议到任务攻击”的完整链路。^[inferred]
- 若往前追溯方法脉络，可把 [[concepts/Indirect Local Attack in Segmentation]] 看作“上下文脆弱性”证据，把 [[concepts/SegPGD]] 看作“白盒强攻击/强训练基线”。

## 可复用模式

- 先定义代理指标（如 mIoU）保证更新方向一致。
- 将连续优化问题变成离散可控搜索。
- 在固定预算下报告攻击退化曲线与低查询性能。

## Open Questions

- 如何扩展到 targeted segmentation attack？
- 如何与 foundation segmentation models 对齐评估口径？
- 是否可引入视觉可感知性的人类评测协议？

## 关联

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Discrete Linear Noise]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[entities/Cityscapes Dataset]]
- [[entities/ADE20K Dataset]]
- [[projects/index]]

## 联网补充

- 《Delving into Decision-based Black-box Attacks on Semantic Segmentation》提出 DLA，说明在仅有决策反馈时依然可以高效破坏分割性能，查询预算是核心评测轴。
- 该方向提示分割鲁棒性评测应从“是否被攻破”扩展到“在给定查询/扰动预算下的退化曲线”。
- 因而这条脉络可粗分为三步：先由 [[concepts/Indirect Local Attack in Segmentation]] 证明上下文会放大局部扰动，再由 [[concepts/SegPGD]] 强化白盒评测与训练基线，最后由 DLA 推进到 query-limited decision-based 黑盒设置。^[inferred]

## ?????2026-04-10?

- ?????????????????????????????
- ?????? - 语义分割黑盒攻击难点与图像分类不同，不能直接复用边界攻击直觉。 - 以查询效率为核心的攻击框架更能反映真实部署风险。 - DLA 路线强调三件事：指标稳定、搜索空间压缩、结构化局部校准。
- ????????????
- ??????????? URL???????????????^[ambiguous]
- ????????????arXiv / DOI / ???????????????????
- ?????????????????????

