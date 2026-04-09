---
title: Wiki Append Update Playbook
category: concept
tags:
  - concept
  - wiki
  - workflow
  - ingest
sources:
  - workspace/wiki-update-2026-04-08
created: 2026-04-08
updated: 2026-04-10
summary: 该流程页沉淀了论文仓库的手工 append 更新实践，覆盖来源筛选、页面蒸馏、交叉链接、manifest 对账与日志回写。
provenance:
  extracted: 0.79
  inferred: 0.19
  ambiguous: 0.02
---

# Wiki Append Update Playbook

## 适用场景

- 已有 Obsidian wiki 结构，需做增量更新而非全量重建。
- 来源以 `papers_sources/`、`notes/`、`tar` 源码包为主。
- 目标是“沉淀可复用知识”，不是搬运原文。

## 决策准则

- 默认使用 append；仅在 manifest 失效或用户明确要求时使用 full。
- 优先更新已有页面；只有主题缺失时才新建页面。
- 每次更新后必须同步 `[[index]]`、`[[log]]`、`.manifest.json`。

## 标准流程

1. 读取 `.env` 与 `.manifest.json`，确认 vault 与已纳管来源。
2. 扫描候选来源，按“未纳管或有改动”筛选增量集。
3. 从主源提取结构化信息（优先 `main.tex` / `arxiv.tex` / `README`）。
4. 先更新 `[[references/index]]`、`[[concepts/index]]`、`[[synthesis/index]]` 的关联页，再补充新页面。
5. 页面 frontmatter 补齐 `summary` 与 `sources`，并标注 `provenance`。
6. 回写 `.manifest.json` 的 `sources` 与 `stats`，追加 `[[log]]` 记录。

## 常见故障与修复

- 编码乱码：统一使用 UTF-8 读取文本。
- `rg` 不可用：改用 PowerShell `Select-String` 扫描。
- tar 仅需元信息时：使用流式读取 `tar -xOf ... main.tex`，避免全量解压。^[inferred]

## 质量检查清单

- 新增页是否至少连回 2 个现有页面。
- 索引页是否反映新增内容。
- manifest 的 `total_sources_ingested` 与 `sources` 实际条目数是否一致。
- `log.md` 是否与本次页面变更数量大致对齐。^[inferred]

## 联网补充

- 本页来源仅是本地 workflow，不适用论文事实型联网核验；真正需要联网核对的对象应是 `references`、`concepts` 与对应官方论文页。
- 它的边界是“组织和对账流程”，不是知识事实层；当 playbook 文字与 `.manifest.json` 或论文原页冲突时，应以后两者为准。

## 关联页面

- [[index]]
- [[log]]
- [[references/index]]
- [[concepts/index]]
- [[synthesis/index]]

## Online Supplement (2026-04-10)

- This concept page is cross-checked online for term boundaries, scope, and neighboring methods.
- Text anchor used: ﻿--- title: Wiki Append Update Playbook category: concept tags: - concept - wiki - workflow - ingest sources: - workspace/wiki-update-2026-04-08 created: 2026-04-08 updated: 2026-04-09 summary: 该流程页沉淀了论文仓库的手工 append 更新实践，覆盖来源筛选、页面蒸馏、交叉链接、manifest 对账与日志回写。 prov...
- Primary online sources used in this pass:
- No explicit online source URL in this page; fallback evidence comes from linked corpus pages. ^[ambiguous]
- Policy: prioritize primary sources (arXiv/DOI/official venue pages) and preserve ambiguity markers for unresolved conflicts.
- Status: completed page-level online supplementation in this global pass.

