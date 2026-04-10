---
title: "Prompt-Robust Representation Probing"
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
title: Prompt-Robust Representation Probing
category: concept
tags:
  - concept
  - llm
  - probing
  - robustness
sources:
  - notes/Language Models Represent Space and Time.md
created: 2026-04-08
updated: 2026-04-10
summary: 通过多提示模板一致性检验，验证探针解码结果不依赖单一 prompt 偶然性。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# Prompt-Robust Representation Probing

重点不在单次探针得分，而在不同提示扰动下是否仍保持稳定可解码结构。

## 联网补充

- 时空表示论文真正有价值的地方在于做了 prompt 变体、大小写扰动和跨实体泛化，避免把单一模板 probe 当成表示证据。
- 但 probe 成功依然不等于模型在生成时因果使用了这些表示，所以它更适合作为“表征存在性”证据，而不是能力归因终点。

## 关联页面

- [[references/Language Models Represent Space and Time]]
- [[concepts/Linear Probe World-Model Evaluation]]
- [[synthesis/Temporal Structure Learning in Sequence Models]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Prompt-Robust Representation Probing category: concept tags: - concept - llm - probing - robustness sources: - notes/Language Models Represent Space and Time.md created: 2026-04-08 updated: 2026-04-08 summary: 通过多提示模板一致性检验，验证探针解码结果不依赖单一 prompt 偶然性。...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

