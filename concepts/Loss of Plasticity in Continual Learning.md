---
title: "Loss of Plasticity in Continual Learning"
category: concept
tags:
  - concept
sources:
  - workspace/wiki-update-2026-04-10-global-lint-remediation
created: 2026-04-10
updated: 2026-04-10
summary: "﻿---"
---
---
title: Loss of Plasticity in Continual Learning
category: concept
tags:
  - concept
  - continual-learning
  - optimization
  - representation
sources:
  - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv.tex
created: 2026-04-08
updated: 2026-04-10
summary: 可塑性丧失指模型在持续学习中逐步失去学习新任务的能力，区别于灾难性遗忘，需单独监测与干预。
provenance:
  extracted: 0.9
  inferred: 0.08
  ambiguous: 0.02
---

# Loss of Plasticity in Continual Learning

## 定义

模型在持续任务流中逐步降低“继续学习新任务”的能力，即使不是主要由遗忘旧任务引起。

## 与灾难性遗忘区别

- 灾难性遗忘：旧知识保持失败。
- 可塑性丧失：新知识吸收能力衰退。
- 两者可同时存在，但成因与修复路径不完全相同。

## 观测信号

- 长任务序列表现持续下滑。
- 死单元比例上升、权重幅度扩大、有效秩下降。

## 联网补充

- Nature 摘要直接给出结论：标准深度学习方法在 continual-learning 设置下会逐渐失去 plasticity，最后学新任务的能力接近浅层网络。
- 这比 catastrophic forgetting 更根本，因为问题不只是“记不住旧任务”，而是“连新目标都越来越难优化”。

## 关联页面

- [[references/Maintaining Plasticity in Deep Continual Learning]]
- [[concepts/Continual Backpropagation]]
- [[concepts/Feature Effective Rank Diagnostics]]
- [[concepts/Selective Neuron Reinitialization]]
- [[synthesis/Continual Learning Plasticity Maintenance Playbook]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Loss of Plasticity in Continual Learning category: concept tags: - concept - continual-learning - optimization - representation sources: - papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv.tex created: 2026-04-08 updated: 2026-...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

