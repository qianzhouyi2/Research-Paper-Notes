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
summary: "跨论文总结：鲁棒语义分割的可靠评测必须同时关注攻击强度、指标口径和训练协议；SEA 是当前该方向的重要校准节点。"
provenance:
  extracted: 0.79
  inferred: 0.18
  ambiguous: 0.03
---

# Reliability and Benchmarking for Robust Segmentation

## 核心结论

- 语义分割鲁棒评测不能只看某一个攻击下的 pixel accuracy，也不能只看单一 loss 的 PGD 结果。
- 可靠评测至少要同时报告攻击预算、accuracy、mIoU，以及攻击协议本身的强度。
- [[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models|Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]] 的关键贡献，是把“评测协议本身会系统性高估鲁棒性”这件事用更强攻击实证化。

## 本轮补充（2026-04-16）

- SEA 说明 SegPGD / CosPGD 之类已有强基线在若干模型上仍会低估最坏情况，尤其是在 mIoU 上。
- 该论文还提示一个重要方法论原则：如果训练论文使用的评测协议偏弱，那么所谓“鲁棒提升”可能只是被攻击不够强。
- 目前 SEA 仍是强白盒协议，而不是白盒 + 黑盒混合 benchmark，因此对梯度掩蔽类防御仍应保持谨慎。

## 覆盖论文

- [[references/Benchmarking the Robustness of Semantic Segmentation Models (CVPR 2020)|Benchmarking the Robustness of Semantic Segmentation Models (CVPR 2020)]]
- [[references/Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)|Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)]]
- [[references/Reliability in Semantic Segmentation - Are We on the Right Track|Reliability in Semantic Segmentation: Are We on the Right Track?]]
- [[references/Evaluating the Adversarial Robustness of Semantic Segmentation|Evaluating the Adversarial Robustness of Semantic Segmentation]]
- [[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]]
- [[references/SemSegBench and DetecBench - Benchmarking Reliability and Generalization Beyond Classification|SemSegBench and DetecBench: Benchmarking Reliability and Generalization Beyond Classification]]

## 可复用模式

- 先校准攻击协议，再讨论防御结论。
- 把 accuracy 与 mIoU 分开看，再汇总成同一条 robustness judgment。
- 把训练预算、攻击预算与数据集难度一起报告，避免只比较单一数字。^[inferred]

## 关联概念与链接

- [[concepts/Segmentation Robustness Benchmark Protocol]]
- [[concepts/Standardized Evaluation Attack (SEA) Protocol|Segmentation Ensemble Attack (SEA) Protocol]]
- [[concepts/End-to-End Reliability Reporting for Segmentation Systems]]
- [[concepts/Robustness-Generalization Decoupling in Dense Prediction]]
