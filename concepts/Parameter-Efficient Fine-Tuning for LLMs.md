---
title: "Parameter-Efficient Fine-Tuning for LLMs"
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
title: Parameter-Efficient Fine-Tuning for LLMs
category: concept
tags:
  - concept
  - llm
  - adaptation
  - efficiency
sources:
  - notes/LoRA Low-Rank Adaptation of Large Language Models.md
  - notes/Amortizing intractable inference in large language models.md
created: 2026-04-08
updated: 2026-04-10
summary: 通过限制可训练参数子空间或结构增量，实现低成本微调并保留大模型基础能力。
provenance:
  extracted: 0.8
  inferred: 0.18
  ambiguous: 0.02
---

# Parameter-Efficient Fine-Tuning for LLMs

该范式强调在“训练成本、部署开销、效果保持”三者之间做工程可落地的折中。

## 联网补充

- PEFT 的核心是把任务更新限制在一小块可训练子空间里，LoRA 只是其中最成功的一种低秩实现。
- 它主要优化适配成本、checkpoint 体积和多任务切换，不会自动降低推理期搜索、验证或长链生成的算力开销。

## 关联页面

- [[references/LoRA Low-Rank Adaptation of Large Language Models]]
- [[concepts/Low-Rank Adaptation for LLMs]]
- [[synthesis/Parameter-Efficient LLM Adaptation and Inference]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Parameter-Efficient Fine-Tuning for LLMs category: concept tags: - concept - llm - adaptation - efficiency sources: - notes/LoRA Low-Rank Adaptation of Large Language Models.md - notes/Amortizing intractable inference in large language models.md cr...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

