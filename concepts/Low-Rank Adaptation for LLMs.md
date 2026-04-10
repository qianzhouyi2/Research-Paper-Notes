---
title: "Low-Rank Adaptation for LLMs"
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
title: Low-Rank Adaptation for LLMs
category: concept
tags:
  - concept
  - llm
  - finetuning
  - parameter-efficient
sources:
  - notes/LoRA Low-Rank Adaptation of Large Language Models.md
created: 2026-04-08
updated: 2026-04-10
summary: 低秩适配通过在冻结主干权重上注入小规模低秩增量，实现参数高效微调并降低部署成本。
provenance:
  extracted: 0.9
  inferred: 0.08
  ambiguous: 0.02
---

# Low-Rank Adaptation for LLMs

LoRA 的核心价值是把“全量微调”改写为“低秩增量学习”，减少显存与训练成本。

## 联网补充

- LoRA 冻结底座权重，只学习低秩增量矩阵，并可在推理前合并回原权重，所以它解决的是“训练与部署成本”而不是“推理链长度”。
- 低秩是假设不是公理：秩设得过小、或注入层选得不对时，适配能力会先于参数节省触顶。

## 关联页面

- [[references/LoRA Low-Rank Adaptation of Large Language Models]]
- [[synthesis/LLM Inference Efficiency and Scaling]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Low-Rank Adaptation for LLMs category: concept tags: - concept - llm - finetuning - parameter-efficient sources: - notes/LoRA Low-Rank Adaptation of Large Language Models.md created: 2026-04-08 updated: 2026-04-08 summary: 低秩适配通过在冻结主干权重上注入小规模低秩增量，实...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

