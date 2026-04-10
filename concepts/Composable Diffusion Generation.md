---
title: "Composable Diffusion Generation"
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
title: Composable Diffusion Generation
category: concept
tags:
  - concept
  - diffusion
  - generation
  - compositionality
sources:
  - notes/Compositional Visual Generation with Composable Diffusion Models.md
created: 2026-04-08
updated: 2026-04-10
summary: 可组合扩散生成通过在扩散过程中组合多个条件或子目标，提升复杂语义约束下的可控图像生成能力。
provenance:
  extracted: 0.88
  inferred: 0.1
  ambiguous: 0.02
---

# Composable Diffusion Generation

该范式强调“条件可组合”而非单条件生成，适合复合语义场景。

## 联网补充

- Composable Diffusion 的关键是把多个条件模型的 score / guidance 在采样时组合，因此无需为每种属性组合单独重训模型。
- 它最适合条件近似可因子化的场景；当属性彼此冲突时，组合 guidance 可能牺牲保真度或生成稳定性。

## 关联页面

- [[references/Compositional Visual Generation with Composable Diffusion Models]]
- [[references/Multimodal Chain-of-Thought Reasoning in Language Models]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Composable Diffusion Generation category: concept tags: - concept - diffusion - generation - compositionality sources: - notes/Compositional Visual Generation with Composable Diffusion Models.md created: 2026-04-08 updated: 2026-04-08 summary: 可组合扩...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

