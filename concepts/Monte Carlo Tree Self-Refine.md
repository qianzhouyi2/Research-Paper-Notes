---
title: "Monte Carlo Tree Self-Refine"
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
title: Monte Carlo Tree Self-Refine
category: concept
tags:
  - concept
  - llm
  - reasoning
  - mcts
sources:
  - notes/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report.md
created: 2026-04-08
updated: 2026-04-10
summary: 将 MCTS 与自评估-自修正循环结合，让 LLM 在高难推理中以搜索替代单次采样。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# Monte Carlo Tree Self-Refine

核心思想是把“生成候选、打分、回传、再生成”组织成树搜索闭环，减少一次性 CoT 的路径依赖。

## 联网补充

- MCTSr 把 MCTS 和 self-refine 结合起来，不只扩展候选解，还会把已有中间解作为可重写节点反复修补。
- 这种方法依赖可用的评分或验证信号；没有可靠 value guidance 时，树会变宽，但不一定更接近正确解。

## 关联页面

- [[references/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report]]
- [[concepts/Monte Carlo Tree Search for LLM Reasoning]]
- [[concepts/Tree of Thoughts Reasoning]]
- [[synthesis/LLM Reasoning Search and Verification]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Monte Carlo Tree Self-Refine category: concept tags: - concept - llm - reasoning - mcts sources: - notes/Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B A Technical Report.md created: 2026-04-0...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

