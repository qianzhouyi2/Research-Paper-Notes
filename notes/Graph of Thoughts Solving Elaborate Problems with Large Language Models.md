# Graph of Thoughts: Solving Elaborate Problems with Large Language Models

- 阅读日期：[[2026-04-08]]
- 阅读状态：已读
- 标签：#paper #imported #notion
- 相关方向：结构化推理、图搜索、LLM 规划
- 阅读目的：系统整理 GoT 的图推理范式、系统架构与任务实证

---

## 1. 论文信息

- 题目：Graph of Thoughts: Solving Elaborate Problems with Large Language Models
- 链接：https://arxiv.org/abs/2308.09687
- 作者：Maciej Besta, Nils Blach, Agnieszka Białek, Mirostanislav Henniger, Mateusz Podstawski, Luca Gianinazzi, Ali Hojjat, Julian Giger, Mohammed Cherti, et al.
- 单位：ETH Zurich 等（按论文作者机构）
- 会议 / 期刊 / 年份：arXiv 2023
- 关键词（3~8个）：Graph of Thoughts, reasoning graph, thought transformation, LLM planning, scoring and ranking
- 论文一句话主题：将 LLM 的 thought 组织为任意图结构，通过生成、精炼、聚合等图变换提升复杂任务推理质量。

---

## 2. 先看结论（适合快速回顾）

- 这篇论文主要解决什么问题：CoT/ToT 结构受限，难以表达跨路径融合与反馈循环。
- 提出的核心方法是什么：GoT（Graph of Thoughts）+ 可扩展系统架构（Prompter/Parser/Scorer/Controller + GoO/GRS）。
- 最终最重要的结果是什么：在排序等复杂任务中，质量显著提升且成本可低于 ToT（论文报告排序质量相对 ToT 提升约 62%，成本下降约 31%）。
- 我现在是否值得深入读：值得
- 原因：它不仅提方法，还给出可落地的软件架构与评估方法。

---

## 3. 问题定义

### 3.1 研究问题

- 核心问题：如何让 LLM 推理从“链/树”扩展到“可合并、可回环、可重组”的图结构。
- 输入是什么：任务输入 + 当前推理图状态（thought 节点与依赖边）。
- 输出是什么：最终答案 thought（可来自多路径聚合）。
- 优化目标：在给定成本预算下，提高复杂任务上的解质量与稳定性。
- 任务设定：允许多种 thought transformation（生成、精炼、聚合）并在图上执行。

### 3.2 为什么重要

- 这个问题为什么值得做：复杂任务天然需要多路径试探、比较、整合，单链/单树表达力不足。
- 现实应用价值：适合文档合并、排序、集合操作、关键词统计等可分解任务。
- 学术上的意义：把 prompt engineering 从“模板技巧”提升为“图算法范式”。

### 3.3 难点

- 难点 1：图搜索空间巨大，容易失控。
- 难点 2：thought 质量评估与排序若不可靠，会误导后续扩展。
- 难点 3：不同任务需要不同图拓扑与变换策略，难有单一最优配置。

### 3.4 背景与记号（原始内容归并）

- IO：输入直接映射到输出。
- CoT：引入中间 thought，但通常仍是线性过程。
- CoT-SC：多链采样后选优，仍缺少路径内局部探索。
- ToT：树结构搜索，支持回溯，但难表达跨分支聚合。
- GoT：允许任意图结构，边表示 thought 依赖，可实现跨链融合与反馈。

![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_001.jpg]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_002.png]]

---

## 4. 论文方法

### 4.1 方法总览

- 方法名称：Graph of Thoughts（GoT）
- 一句话概括方法：在图上组织与变换 thought，而不是仅沿链或树展开推理。
- 方法整体流程（按步骤写 1/2/3/4）：
  1. 用 GoO（Graph of Operations）定义静态执行计划。
  2. 在 GRS（Graph Reasoning State）中维护动态 thought 图状态。
  3. 执行 thought transformation（生成/精炼/聚合）并评分排序。
  4. 按控制策略推进到终止并输出最优 thought。

### 4.2 核心设计

#### 设计 1：Reasoning Graph 建模
- 做了什么：把推理过程建模为有向图 `G=(V,E)`，节点是 thought，边是依赖关系。
- 为什么这样设计：支持跨路径共享、回收与重组信息。
- 解决的难点：难点 3。

#### 设计 2：Thought Transformations
- 做了什么：
  - Generation：基于已有 thought 生成新 thought；
  - Refinement：对现有 thought 进行改写增强；
  - Aggregation：聚合多个 thought 形成新 thought。
- 为什么这样设计：把复杂任务分解成可组合操作。
- 解决的难点：难点 1/3。

#### 设计 3：Scoring & Ranking
- 做了什么：对 thought 评分 `s(·)` 并排序 `r(·)`，决定保留/扩展集合。
- 为什么这样设计：控制图膨胀，聚焦高价值路径。
- 解决的难点：难点 1/2。

### 4.3 训练 / 推理细节

- 训练阶段做了什么：框架本身偏推理时方法，无需专门训练。
- 推理阶段做了什么：按 GoO 执行图操作，Prompter/Parser/Scorer/Controller 协同推进。
- 损失函数组成：依任务而定（很多场景用任务特定评分函数）。
- 关键超参数：分支数、图深度、保留候选数、评分阈值。
- 复杂度 / 额外开销：较 CoT 更高，但通过分解与聚合可获得更优质量-成本折中。

### 4.4 系统架构与图变换证据

![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_003.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_004.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_005.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_006.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_007.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_008.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_009.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_010.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_011.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_012.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_013.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_014.png]]

---

## 5. 论文贡献（只写作者真正新增的东西）

- 贡献 1：提出 GoT，将 LLM 推理统一到“任意图结构”范式。
- 贡献 2：提出模块化系统架构，支持快速扩展操作与模型。
- 贡献 3：在多个任务（排序、集合操作、关键词统计、文档合并）上展示优势。
- 贡献 4：提出思想体积（volume of thoughts）指标，量化推理图信息覆盖。

---

## 6. 实验设置

- 任务：
  - Sorting（含重复数字排序）
  - Set Operations（集合交集）
  - Keyword Counting（关键字计数）
  - Document Merging（文档合并）
- 模型：主要使用 GPT-3.5（并报告 Llama-2 观察）
- 对比方法：IO、CoT、CoT-SC、ToT（不同分支/深度配置）
- 评价指标：
  - 任务误差（如排序 error-scope）
  - 成本（调用/token）
  - 延迟与思想体积（volume）分析
- 关键设置：约 100 样本/任务；控制各方法在相近成本条件下比较。

### 6.1 任务示例与评分定义证据

![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_015.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_016.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_017.png]]

---

## 7. 主要结果

### 7.1 主结果

- 结果 1：GoT 相比 ToT 在排序任务上显著降误差，并降低成本（论文报告中位误差约降 62%，成本降 >31%）。
- 结果 2：GoT 相比 IO/CoT 在复杂规模任务上质量提升更明显。
- 结果 3：问题规模越大，GoT 的相对优势越突出。

### 7.2 从结果中能读出的结论

- 结论 1：核心收益来自“任务分解 + 子结果聚合”而非简单增加采样。
- 结论 2：图结构允许“高体积、低延迟”的更优信息流形态。
- 结论 3：当任务可拆分时，GoT 比链式/树式方案更稳健。

### 7.3 最关键的证据

- 最关键表格：不同方法在同成本预算下的质量比较表。
- 最关键图：GoT vs ToT/CoT 的误差与成本曲线、规模扩展分析图。
- 最关键数字：排序任务相对 ToT 的质量提升与成本下降。
- 为什么它最关键：直接验证了“图聚合”带来的实用收益。

### 7.4 结果图（完整）

![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_018.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_019.png]]
![[assets/Graph of Thoughts Solving Elaborate Problems with Large Language Models/image_020.png]]

---

## 8. 消融实验

- 消融点 1：去掉聚合操作后，质量明显下降。
- 消融点 2：仅增加树搜索深度但不做图融合，成本上升且收益受限。
- 消融点 3：评分/排序策略变化会显著影响搜索稳定性。

---

## 9. 和已有工作的关系

- 这篇论文最接近哪些方法：CoT、CoT-SC、ToT、一般图搜索/规划范式。
- 和已有方法相比，最大的不同：允许任意图结构 thought transformation，而非只走链或树。
- 真正的新意在哪里：把 prompt 推理过程抽象为可编排图操作系统。
- 哪些地方更像“工程改进”而不是“方法创新”：具体评分函数与任务提示模板。
- 这篇论文在整个研究脉络里的位置：从“提示技巧”到“推理操作系统”的关键过渡工作。

---

## 10. 我的理解（这一节不能照抄论文）

- 直观理解：GoT 像给 LLM 加了一个“图计算外壳”，让 thought 可以像数据流一样组合与回收。
- 最值得关注的设计：GoO（静态计划）+ GRS（动态状态）二层结构，分离了策略与执行。
- 和已有方法相比的新意：不仅多分支，还支持跨分支融合与循环改进。
- 我认为最强的一点：在复杂任务中同时兼顾质量与成本。

---

## 11. 局限性

- 局限 1：框架实现复杂度高，对工程能力要求较高。
- 局限 2：评分函数质量直接决定搜索上限。
- 局限 3：不同任务需要定制 GoO，自动化程度仍有限。

---

## 12. 对我的启发

- 能直接借鉴的部分：把复杂任务拆成可执行图操作而非单一 prompt。
- 不能直接照搬的部分：评分函数和排序规则需要任务定制。
- 对我当前课题的启发：可把检索、工具调用、验证步骤都建模为 thought 节点。
- 可以尝试的改进方向：学习型控制器、自适应操作图、成本感知动态剪枝。
- 可以作为 baseline / 对比项 / ablation 的部分：CoT/ToT/无聚合 GoT 对比。

---

## 13. 待验证问题

- 问题 1：如何自动学习 GoO 拓扑而非手工编排？
- 问题 2：GoT 与外部工具执行（代码、检索、规划器）的协同最优策略是什么？
- 问题 3：在长上下文代理任务中，thought volume 与最终质量是否仍强相关？

---

## 14. 一句话总结

- GoT 把 LLM 推理升级为可操作的图过程，通过 thought 变换与聚合在复杂任务中实现了比 CoT/ToT 更优的质量-成本平衡。

---

## 15. 快速索引（便于二次回看）

- 核心公式：thought 评分与排序函数、任务误差评分（如 sorting error-scope）。
- 核心图表：架构图（image_004~014）、用例图（image_015~017）、结果图（image_018~020）。
- 最值得复看的章节：Section 3/4（框架与系统）+ Section 7（评估）。
- 复现时最需要注意的点：操作图设计、评分可靠性、成本约束下的分支控制。

### 15.1 整合说明 / 索引

- 原始 Notion 转录内容已拆解到正文 `1~14` 对应章节。
- 全部 20 张图片已按方法/实验/结果语义位置插入。

### 15.2 导入来源与完整性记录

- 源页面 ID：`bbbc33bc-6083-4bff-98e9-225835141809`
- 抓取块数量：`141`
- 未解析块引用：`0`
- 原始 JSON：`notes/_notion_raw/Graph of Thoughts Solving Elaborate Problems with Large Language Models.json`
- 未解析块 ID：无
- 联网校验来源（2026-04-08）：
  - arXiv: https://arxiv.org/abs/2308.09687
  - 代码: https://github.com/spcl/graph-of-thoughts

### 15.3 已完成自检记录

- [x] 原始笔记所有内容已整理到模板正文。
- [x] 图片已全部插入并归位。
- [x] 已联网补充并校正论文核心信息。


## Wiki 关联

- 参考摘要：[[references/Graph of Thoughts Solving Elaborate Problems with Large Language Models|Graph of Thoughts Solving Elaborate Problems with Large Language Models]]
- 概念锚点：[[concepts/Graph of Thoughts Reasoning]]
- 实体锚点：[[entities/Maciej Besta]]
- 综合页面：[[synthesis/Structured Reasoning Methods for LLMs]]
