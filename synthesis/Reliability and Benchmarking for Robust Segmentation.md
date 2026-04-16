---
title: "Reliability and Benchmarking for Robust Segmentation"
category: synthesis
tags:
  - synthesis
  - semantic-segmentation
  - adversarial-robustness
  - evaluation
sources:
  - workspace/wiki-update-2026-04-10-cn-localization
  - papers_sources/semantic_segmentation_robustness_20260409/download_report.json
  - https://arxiv.org/abs/1908.05005
  - https://arxiv.org/abs/2303.11298
  - https://arxiv.org/abs/2306.12941
  - https://arxiv.org/abs/2306.14217
created: 2026-04-10
updated: 2026-04-16
summary: "跨论文总结：鲁棒语义分割的可靠评测不仅要看攻击强度和指标口径，还要先确认评测协议足够强；否则训练结论本身就不可信。"
provenance:
  extracted: 0.79
  inferred: 0.18
  ambiguous: 0.03
---

# Reliability and Benchmarking for Robust Segmentation

## 核心结论

- 语义分割鲁棒评测不能只看某一个攻击下的 pixel accuracy，也不能只看单一 loss 的 PGD 结果。
- 可靠评测至少要同时报告攻击预算、accuracy、mIoU，以及攻击协议本身的强度。
- 如果评测协议偏弱，那么后续所有“训练更鲁棒了”的结论都可能只是被打得不够狠。
- [[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models|Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]] 的关键贡献，是把“评测协议本身会系统性高估鲁棒性”这件事用更强攻击实证化。

## 本轮补充（2026-04-16）

- SEA 说明 SegPGD / CosPGD 之类已有强基线在若干模型上仍会低估最坏情况，尤其是在 mIoU 上。
- 该论文还提示一个重要方法论原则：如果训练论文使用的评测协议偏弱，那么所谓“鲁棒提升”可能只是被攻击不够强。
- SEA 对 mIoU 的 worst-case 汇总并不是精确全局优化，而是因为全局 mIoU 组合过于复杂，所以采用 greedy 近似；这也提醒评测报告要把协议细节写清楚。
- 目前 SEA 仍是强白盒协议，而不是白盒 + 黑盒混合 benchmark，因此对梯度掩蔽类防御仍应保持谨慎。

## 覆盖论文

- [[references/Benchmarking the Robustness of Semantic Segmentation Models (CVPR 2020)|Benchmarking the Robustness of Semantic Segmentation Models (CVPR 2020)]]
- [[references/Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)|Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)]]
- [[references/Reliability in Semantic Segmentation - Are We on the Right Track|Reliability in Semantic Segmentation: Are We on the Right Track?]]
- [[references/Evaluating the Adversarial Robustness of Semantic Segmentation|Evaluating the Adversarial Robustness of Semantic Segmentation]]
- [[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]]
- [[references/SemSegBench and DetecBench - Benchmarking Reliability and Generalization Beyond Classification|SemSegBench and DetecBench: Benchmarking Reliability and Generalization Beyond Classification]]

## 可复用模式

- 先校准攻击协议，再讨论任何训练或防御结论。
- 把 accuracy 与 mIoU 分开看，再汇总成同一条 robustness judgment。
- 把训练预算、攻击预算与数据集难度一起报告，避免只比较单一数字。^[inferred]
- 明确说明 strongest protocol 的覆盖范围，例如“仅白盒”还是“白盒 + 黑盒”。^[inferred]

## 关联概念与链接

- [[concepts/Segmentation Robustness Benchmark Protocol]]
- [[concepts/Standardized Evaluation Attack (SEA) Protocol|Segmentation Ensemble Attack (SEA) Protocol]]
- [[concepts/End-to-End Reliability Reporting for Segmentation Systems]]
- [[concepts/Robustness-Generalization Decoupling in Dense Prediction]]
