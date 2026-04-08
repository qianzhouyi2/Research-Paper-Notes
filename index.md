---
title: Wiki 首页
aliases:
  - Home
category: index
tags:
  - index
  - wiki
updated: 2026-04-08
summary: 论文阅读 wiki 总入口，汇总导航、工作流与最近 ingest 结果。
---

# 论文阅读 Wiki

这是你的论文阅读知识库入口。

## 快速入口

- [[notes/index|Notes 总览]]
- [[references/index|References（单篇摘要）]]
- [[concepts/index|Concepts（可复用概念）]]
- [[synthesis/index|Synthesis（跨论文总结）]]
- [[projects/index|Projects（专题项目）]]
- [[entities/index|Entities（作者/机构/模型）]]
- [[journal/index|Journal（阅读日志）]]
- [[log|系统日志]]

## 当前资产

- 已导入论文笔记：26 篇（见 [[notes/index]]）
- References：26 篇（含 1 个 MeanSparse 命名桥接页）
- Concepts：35 篇（方法级与协议级概念）
- Entities：45 篇（作者/模型/数据集）
- Synthesis：10 篇（跨论文主题综合）
- 已纳管来源：19 个（见 [[log]] 与 `.manifest.json`）
- 原始资料：[[papers_sources]]
- 模板：[[Templates/论文阅读模板|论文阅读模板]]

## 最新 Ingest（2026-04-08）

- 来源：`papers_sources`（mode=`append`，手工 ingest）
- 新增页：4 个 references、5 个 concepts、1 个 synthesis
- 更新页：[[references/index]]、[[concepts/index]]、[[synthesis/index]]、[[index]]、[[log]]
- 说明：本次对未纳管源码目录与 tar 包执行统一入库，并补充跨页链接

## 最新 Wiki Update（2026-04-08）

- 类型：`wiki-update`（知识沉淀）
- 新增页：21 个 references、8 个 concepts、9 个 entities、3 个 synthesis
- 更新页：[[references/index]]、[[concepts/index]]、[[entities/index]]、[[synthesis/index]]、[[notes/index]]、[[index]]、[[log]]
- 说明：完成 notes 全量蒸馏补齐与 cross-link 收口，当前 `notes` 已全部具备对应 `references`；覆盖度盘点见 [[journal/Wiki Coverage Audit 2026-04-08]]

## 最新 Coverage 补齐（2026-04-08）

- 新增页：4 个 concepts、15 个 entities、1 个 synthesis
- note 级补链：25 篇笔记全部新增 `Wiki 关联` 段
- 当前缺口：`reference` 层 `concept/entity/synthesis` 缺口均为 0

## 最新 Papers Sources 深度细化（2026-04-08）

- 范围：按 `papers_sources` 逐篇细化 9 篇论文（Are aligned、Delving、Hyena、Maintaining Plasticity、MeanSparse、CIF、GPT-ST、Language Models Represent Space and Time、PanGu-π）
- 新增页：12 个 concepts、13 个 entities、4 个 synthesis、1 个 journal audit
- 更新页：9 个 references、8 个已有 concepts、4 个已有 synthesis、9 篇相关 notes 的 `Wiki 关联`
- 盘点入口：[[journal/Papers Sources Deep Gap Audit 2026-04-08]]

## 推荐工作流

1. 把原始 PDF/源码/草稿放入 [[papers_sources]]
2. 在 [[notes]] 产出单篇阅读笔记（建议用模板）
3. 将单篇结论沉淀到 [[references/index]]
4. 抽取可复用方法、术语、公式到 [[concepts/index]]
5. 在 [[synthesis/index]] 做跨论文对比和总结

## 本周可做

- [ ] 为每篇已读笔记至少提炼 3 个概念页
- [ ] 在 `references/` 补齐单篇摘要卡
- [ ] 新建一个 `projects/` 专题页（例如“长上下文建模”）
