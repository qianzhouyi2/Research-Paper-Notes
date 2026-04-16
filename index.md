---
title: Wiki 首页
aliases:
  - Home
category: index
tags:
  - index
  - wiki
sources:
  - workspace/wiki-update-2026-04-10-lint-update-online
  - workspace/wiki-update-2026-04-10-global-lint-remediation
  - workspace/wiki-update-2026-04-10-cn-placeholder-remediation
  - workspace/wiki-update-2026-04-16-towards-reliable-eval-sync
created: 2026-04-08
updated: 2026-04-16
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
- [[_meta/taxonomy|标签词表]]

## 最新 Wiki 更新（2026-04-16，Reliable Eval / PIR-AT Deepening）

- 类型：`wiki-update`（single-paper deepening）。
- 新增页面：[[entities/Francesco Croce]]、[[entities/Naman D. Singh]]。
- 更新页面：[[notes/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]]、[[references/Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models]]、[[concepts/Standardized Evaluation Attack (SEA) Protocol|Segmentation Ensemble Attack (SEA) Protocol]]、[[concepts/Prior-informed Robust Adversarial Training (PIR-AT)]]、[[synthesis/Reliability and Benchmarking for Robust Segmentation]]、[[synthesis/Robust Training Strategies for Semantic Segmentation]]、[[entities/index]]、[[index]]、[[log]]。
- 说明：把当前打开的鲁棒语义分割论文从模板 note 补成完整阅读卡，并把 SEA / PIR-AT / 作者实体正式接入现有知识图谱。

## 最新 Wiki 更新（2026-04-10，内部页中文化）

- 类型：`wiki-update`（concept/entity/synthesis 内部页中文化）。
- 更新页面：97 个（本轮新增且创建日期为 2026-04-10 的 `concepts`、`entities`、`synthesis` 页面）。
- 说明：正文模板统一改为中文，保留标题与论文名原文，清理模板英文段落与异常占位符。

## 最新 Wiki 更新（2026-04-10，Global Lint Remediation）

- 类型：`wiki-lint + wiki-update`（全库历史遗留问题分批修复）。
- 新增页面：[[journal/2026-04-08]]、[[journal/2026-04-06]]。
- 修复内容：补齐 68 个历史页面 frontmatter，修复日期锚点、模板链接、索引漏收录与遗留坏链。
- 结果：全库 lint 已归零（`orphans=0`、`broken_total=0`、`missing_frontmatter=0`、`index_issues=0`、`stale_by_date=0`）。

## 最新 Wiki 更新（2026-04-10，Lint + Online Metadata Calibration）

- 类型：`wiki-lint + wiki-update`（联网核验与元数据校准）。
- 新增页面：[[papers_sources/index]]。
- 更新页面：[[references/Benchmarking the Robustness of Semantic Segmentation Models (CVPR 2020)]]、[[references/Benchmarking the Robustness of Semantic Segmentation Models (IJCV 2020)]]、[[references/On the Robustness of Semantic Segmentation Models to Adversarial Attacks]]、[[references/Towards Semantically Stealthy Adversarial Attacks Against Segmentation Models]]、[[references/RP-PGD - Enhancing Semantic Segmentation Robustness via Region-based Prioritized PGD]]、[[references/SemSegBench and DetecBench - Benchmarking Reliability and Generalization Beyond Classification]]、[[references/Erosion Attack for Adversarial Training to Enhance Semantic Segmentation Robustness (arXiv preprint)]]、[[references/Evaluating the Adversarial Robustness of Semantic Segmentation]]、[[references/Proximal Splitting Adversarial Attack for Semantic Segmentation]]、[[references/Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement]]、[[references/Uncertainty-Based Detection of Adversarial Attacks in Semantic Segmentation]]、[[index]]、[[log]]。
- 结果：语义分割鲁棒性子集 lint 硬检查已清零（`orphans=0`、`broken=0`、`missing_frontmatter=0`）。

## 最新 Wiki 更新（2026-04-09，NES + L-infinity Calibration）

- 类型：`wiki-update`（concept + source-note calibration）
- 新增页面：[[concepts/Natural Evolutionary Strategies (NES)]]、[[concepts/L-infinity Norm Ball]]
- 更新页面：[[concepts/Discrete Linear Noise]]、[[concepts/Decision-based Black-box Attack for Segmentation]]、[[concepts/Proxy Index mIoU Optimization]]、[[concepts/Query-Efficient Attack Evaluation]]、[[concepts/index]]、[[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]、[[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]、[[papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh]]、[[index]]、[[log]]
- 说明：把这轮对话里解释过的 Random Attack、NES 和 `L_\infty` 范数球沉淀进语义分割攻击主线，并联网校准了“边界”与“顶点”这两个容易混淆的说法。

## 最新 Wiki 更新（2026-04-09，Segmentation Attack Priors）

- 类型：`wiki-update`（concept distillation）
- 新增页面：[[concepts/Indirect Local Attack in Segmentation]]、[[concepts/SegPGD]]
- 更新页面：[[concepts/Decision-based Black-box Attack for Segmentation]]、[[synthesis/Decision-based Segmentation Attack Landscape]]、[[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]、[[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]、[[concepts/index]]、[[index]]、[[log]]
- 说明：补齐语义分割攻击主线里的两类前置方法锚点：ILA 解释“局部扰动为何会借长程上下文影响远处区域”，SegPGD 概括“如何用动态像素加权提升白盒攻击效率”。

## 最新 Wiki 更新（2026-04-09，ViT + Black-box Online Refresh）

- 类型：`wiki-update`（concept online refresh）
- 新增页面：[[concepts/Vision Transformer (ViT)]]、[[concepts/SignSGD]]、[[concepts/SimBA (Simple Black-box Attack)]]、[[concepts/Square Attack]]
- 更新页面：[[concepts/Query-Efficient Attack Evaluation]]、[[concepts/index]]、[[index]]、[[log]]
- 说明：对 ViT / 黑盒攻击 / SignSGD / SimBA / Square Attack 做了联网核验并补充时间锚点与方法边界。

## 当前资产

- 已导入论文笔记：27 篇（见 [[notes/index]]）
- References：54 篇
- Concepts：90 篇
- Entities：133 篇
- Synthesis：29 篇
- 已纳管来源：49 个（见 [[log]] 与 `.manifest.json`）
- 原始资料：[[papers_sources/index|papers_sources]]
- 模板：`Templates`

## 最新 Wiki 更新（2026-04-09，Segmentation Robustness Corpus Sync）

- 类型：`wiki-update`（corpus sync）
- 新增页面：[[synthesis/Semantic Segmentation Robustness Corpus 2019-2026]]、[[journal/Semantic Segmentation Robustness Corpus Update 2026-04-09]]
- 更新页面：[[synthesis/index]]、[[journal/index]]、[[index]]、[[log]]
- 说明：将 `papers_sources/semantic_segmentation_robustness_20260409` 的 28 篇语义分割鲁棒性论文下载结果同步入库（22 个 TeX 源码包 + 6 个 PDF 回退）。

## 最新 Wiki 更新（2026-04-09，模型实体补齐）

- 类型：`wiki-update`（entity 增补）
- 新增页：4 个 entities（[[entities/FCN]]、[[entities/PSPNet]]、[[entities/DeepLabv3]]、[[entities/MaskFormer]]）
- 更新页：[[entities/SegFormer]]、[[entities/index]]、[[concepts/Semantic Segmentation]]、[[concepts/Query-Efficient Attack Evaluation]]、[[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]、[[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]、[[index]]、[[log]]
- 说明：补齐分割评测五模型实体锚点（FCN/PSPNet/DeepLabv3/SegFormer/MaskFormer），并把 note/reference/concept 的模型链接统一回链到实体层。

## 上一轮 Wiki 更新（2026-04-09）

- 类型：`wiki-update`（concept 增补）
- 新增页：2 个 concepts（[[concepts/Deep Neural Network (DNN)]]、[[concepts/Semantic Segmentation]]）
- 更新页：[[concepts/index]]、[[papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh]]、[[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]、[[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]、[[index]]、[[log]]
- 说明：新增 DNN 与 Semantic Segmentation 两个上位概念卡，并把源文、note、reference 的语义分割锚点统一回链到概念层。

## 最新 Ingest（2026-04-08）

- 来源：`papers_sources`（mode=`append`，手工 ingest）
- 新增页：4 个 references、5 个 concepts、1 个 synthesis
- 更新页：[[references/index]]、[[concepts/index]]、[[synthesis/index]]、[[index]]、[[log]]
- 说明：本次对未纳管源码目录与 tar 包执行统一入库，并补充跨页链接

## 最新 Wiki 更新（2026-04-08）

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

## 最新 Notes 细扫补链（2026-04-08）

- 类型：`wiki-update`（notes 精细化补链）
- 新增页：5 个 concepts、3 个 entities、3 个 synthesis
- note 级更新：20 篇笔记增强 `Wiki 关联`（重点补齐 reasoning / multi-agent / multimodal / efficiency 四类主题）
- 说明：从“覆盖到位”提升到“语义更细”，新增 `MCTS Self-Refine`、`Process-Supervised Step Verification`、`Mixture-of-Agents Aggregation` 等锚点页

## 最新 Notes 细扫补链（第二轮，2026-04-08）

- 类型：`wiki-update`（notes 深层语义补链）
- 新增页：6 个 concepts、4 个 entities、1 个 synthesis
- note 级更新：9 篇笔记增强 `Wiki 关联`（重点补齐概率推理、软对齐、多模态两阶段推理与 PEFT 主题）
- 说明：新增 `Amortized Bayesian Inference for LLMs`、`GFlowNet Posterior Sampling for Text Generation`、`Soft Monotonic Alignment`、`Two-Stage Multimodal CoT` 等概念锚点，并新增综合页 [[synthesis/Probabilistic Inference-Time Control for LLMs]]

## 最新 Notes 细扫补链（第三轮，2026-04-08）

- 类型：`wiki-update`（notes 深层语义补链）
- 新增页：4 个 concepts、12 个 entities、5 个 synthesis
- note 级更新：14 篇笔记增强 `Wiki 关联`（重点补齐 complexity labeling、CoT removal smoothing、时序结构学习与作者/数据集实体）
- 说明：新增 [[synthesis/Temporal Structure Learning in Sequence Models]]、[[synthesis/Process Supervision and CoT Internalization]]、[[synthesis/Search-Based Deliberate Reasoning Landscape]]、[[synthesis/Inference-Time Orchestration and Routing for LLMs]]、[[synthesis/Robust Representation and Adversarial Dynamics]]，并将 23 篇笔记提升为 `synthesis>=3`

## 最新 Notes 细扫补链（第四轮，2026-04-08）

- 类型：`wiki-update`（notes 主题综合收口）
- 新增页：2 个 synthesis
- note 级更新：5 篇笔记增强 `Wiki 关联`（重点收口 LoRA / Maintaining / PanGu-π / MeanSparse / Language 表示容量主题）
- 说明：新增 [[synthesis/Efficient Adaptation and Plasticity Retention]] 与 [[synthesis/Representation Capacity and Effective Rank]]，当前 25 篇笔记已全部达到 `concepts>=3`、`entities>=3`、`synthesis>=3`

## 最新 联网核验更新（2026-04-08）

- 类型：`wiki-update`（online metadata refresh）
- 新增页：10 个 entities
- 清理页：移除 7 个误归因 author entities
- 页面修正：6 篇 notes、26 个 references 完成作者 / venue / source 核验
- 覆盖结果：26/26 `reference` 卡现已全部带有 `联网核验` 条目
- 说明：前半轮重点修正 CIF、From Explicit CoT、GPT-ST、Math-Shepherd、Mixture-of-Agents、Synergy-of-Thoughts 的作者归因；后半轮补齐其余 20 个 reference 卡的官方来源核验。`concepts` 与 `synthesis` 层复检后未发现需要基于联网事实改写的漂移项

## 最新 Concepts 联网补充（2026-04-08）

- 类型：`wiki-update`（online concept refresh）
- 覆盖：50/50 `concepts` 完成联网复核
- 页面更新：27 个轻量 concept 卡新增 `联网补充` 段；无新增结构页
- 主题重点：budget routing、search-based reasoning、multimodal CoT、PEFT、CIF、multi-agent orchestration
- 说明：本轮依据官方论文页与已整理 notes 补足机制定义、适用边界与跨论文区分；其余 23 个 concept 页结构已足够，仅完成复核不做机械改写

## 最新 Concepts 联网补充（第二轮，2026-04-09）

- 类型：`wiki-update`（online concept refresh）
- 覆盖：剩余 23/23 `concepts` 补齐 `联网补充`
- 页面更新：22 个学术 concept 页 + 1 个本地 workflow 页；无新增结构页
- 主题重点：segmentation attacks、adversarial alignment、Hyena、plasticity、GPT-ST、Language world-model probes、CIF
- 累计结果：52/52 `concepts` 页面现已全部带有 `联网补充`

## 最新 Entities 联网补充（2026-04-09）

- 类型：`wiki-update`（online entity refresh）
- 覆盖：71/71 `entities` 补齐 `联网补充`
- 页面更新：52 个 author、13 个 model、5 个 dataset、1 个 benchmark；无新增结构页
- 主题重点：作者归因、一手模型定义、官方数据集范围、外部 benchmark 角色
- 说明：本轮把实体层从“名称占位卡”提升到“实体定义 + 在本库中的作用”双层结构，便于后续继续 ingest 时直接复用

## 最新 Synthesis 联网补充（2026-04-09）

- 类型：`wiki-update`（online synthesis refresh）
- 覆盖：21/21 `synthesis` 补齐 `联网补充`
- 页面更新：21 个主题综合页；无新增结构页
- 主题重点：鲁棒评测阶梯、结构化推理搜索、推理时路由编排、参数高效适配、时空结构学习
- 说明：本轮把综合层统一升级为“主题结论 + 联网补充”双层结构，便于后续跨论文对照时快速引用外部事实锚点

## 推荐工作流

1. 把原始 PDF/源码/草稿放入 [[papers_sources/index|papers_sources]]
2. 在 [[notes/index|notes]] 产出单篇阅读笔记（建议用模板）
3. 将单篇结论沉淀到 [[references/index]]
4. 抽取可复用方法、术语、公式到 [[concepts/index]]
5. 在 [[synthesis/index]] 做跨论文对比和总结

## 本周可做

- [x] 为重点薄弱笔记补齐第二轮概念/实体/综合锚点
- [x] 在 `references/` 补齐单篇摘要卡并完成联网核验
- [ ] 新建一个 `projects/` 专题页（例如“长上下文建模”）

## Latest Wiki Ingest (2026-04-10, Segmentation Robustness Batch 1-4)

- Type: `wiki-ingest + wiki-update` (batch-wise per-paper verification).
- Created: 28 references, 30 concepts, 60 entities, 6 synthesis, 1 journal page.
- Updated: [[references/index]], [[concepts/index]], [[entities/index]], [[synthesis/index]], [[journal/index]], [[index]], [[log]].
- Note: all 28 papers were checked against online metadata (arXiv/Crossref) and local downloads.
- Follow-up: completed one-by-one online supplement pass for all extracted concepts/entities/synthesis pages.
- Deepening pass: upgraded all 28 reference cards with local abstract evidence and added [[synthesis/Segmentation Robustness Batch Reading Matrix 2019-2026]].

- [[_meta/taxonomy|标签词表]]

## 最新全库中文修复与检查（2026-04-10）

- 类型：`wiki-update + wiki-lint`（全库修复）
- 修复范围：153 个 `concepts/entities/synthesis` 页面移除异常占位尾块；并修正首页 `notes` 导航链接。
- 校验结果：`workspace/wiki_lint_global_2026-04-10_cnfix_final.json` 显示 `orphans=0`、`broken_links=0`、`missing_frontmatter=0`、`missing_summary=0`、`placeholder_like_issues=0`。
- 说明：本轮重点是把内部页残留的问号占位内容彻底清理，保留既有标题与知识结构，仅修复页面可读性和一致性。
