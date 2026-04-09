---
title: Multi-Agent LLM Collaboration Landscape
category: synthesis
tags:
  - synthesis
  - llm
  - multi-agent
  - collaboration
sources:
  - notes/Chain of Agents Large Language Models Collaborating on Long-Context Tasks.md
  - notes/Mixture-of-Agents Enhances Large Language Model Capabilities.md
created: 2026-04-08
updated: 2026-04-10
summary: 多智能体 LLM 协作通过角色分工与聚合机制提升任务质量，主要权衡点在于推理成本、延迟与编排复杂度。
provenance:
  extracted: 0.78
  inferred: 0.2
  ambiguous: 0.02
---

# Multi-Agent LLM Collaboration Landscape

## 关键范式

- 链式协作（Chain of Agents）：强调中间结果传递和长上下文稳态处理。
- 混合聚合（Mixture-of-Agents）：强调多模型并行提案与层次聚合。

## 主要收益

- 降低单模型在长链任务中的上下文丢失风险。
- 通过多视角提案提高答案稳定性与覆盖度。^[inferred]

## 主要约束

- 计算成本和延迟上升。
- 编排策略和角色定义敏感。^[inferred]

## 代表页面

- [[references/Chain of Agents Large Language Models Collaborating on Long-Context Tasks]]
- [[references/Mixture-of-Agents Enhances Large Language Model Capabilities]]
- [[concepts/Multi-Agent LLM Orchestration]]
- [[synthesis/Inference-Time Orchestration and Routing for LLMs]]

## 联网补充

- Chain-of-Agents 通过 agent 链式分工处理长上下文信息聚合，说明协作式上下文压缩可改善长输入任务表现。
- Mixture-of-Agents 采用分层聚合不同模型输出，体现“异构代理集成”可提升质量，但系统复杂度和调度成本同步上升。

## Online Supplement (2026-04-10)

- This synthesis page is cross-checked online for cross-paper consistency and evaluation-scope alignment.
- Text anchor used: - 链式协作（Chain of Agents）：强调中间结果传递和长上下文稳态处理。 - 混合聚合（Mixture-of-Agents）：强调多模型并行提案与层次聚合。
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

