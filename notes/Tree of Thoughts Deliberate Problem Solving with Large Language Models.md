# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

- 阅读日期：[[2026-04-08]]
- 阅读状态：已读
- 标签：#paper #imported #notion
- 相关方向：LLM 推理、搜索增强推理、规划与回溯
- 阅读目的：系统整理 ToT 框架的方法细节、实验设置与可复用结论

---

## 1. 论文信息

- 题目：Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- 链接：https://arxiv.org/abs/2305.10601
- 作者：Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
- 单位：Princeton University / Google Research（按论文作者机构归并）
- 会议 / 期刊 / 年份：NeurIPS 2023（arXiv 2023）
- 关键词（3~8个）：Tree of Thoughts, deliberate reasoning, BFS/DFS search, self-evaluation, CoT extension
- 论文一句话主题：把 LLM 推理从“单链生成”升级为“树搜索 + 自评估 + 回溯”的通用问题求解框架。

---

## 2. 先看结论（适合快速回顾）

- 这篇论文主要解决什么问题：线性 CoT 只做单路径左到右展开，难以探索、规划与回溯。
- 提出的核心方法是什么：Tree of Thoughts（ToT）框架，包含 thought 分解、thought 生成、状态评估和搜索算法四个模块。
- 最终最重要的结果是什么：在 Game of 24、Creative Writing、Mini Crosswords 三类任务上显著优于 IO/CoT 等基线。
- 我现在是否值得深入读：值得
- 原因：方法通用、模块化、无需额外训练，且对复杂推理任务有直接增益。

---

## 3. 问题定义

### 3.1 研究问题

- 核心问题：如何让 LLM 在中间推理步骤上进行多分支探索和全局决策，而不是只沿单条链前进。
- 输入是什么：任务输入 `x`，中间 thought 序列 `z_1...z_i`。
- 输出是什么：最终解 `y`（或结构化结果）。
- 优化目标：在有限预算下，找到更高可行性/更高质量的解路径。
- 任务设定：以自然语言 thought 作为中间状态，支持局部扩展与全局搜索。

### 3.2 为什么重要

- 这个问题为什么值得做：很多任务（规划、组合搜索、创作）本质是非单路径决策问题。
- 现实应用价值：能提高复杂任务成功率并增强可解释的中间推理过程。
- 学术上的意义：把 LLM 推理明确建模为“启发式搜索”而不是纯解码。

### 3.3 难点

- 难点 1：如何定义可操作、可扩展的 thought 单元。
- 难点 2：如何对部分解状态进行启发式评价（而非只看最终答案）。
- 难点 3：如何在成本可控下实现前瞻、剪枝与回溯。

### 3.4 背景与形式化（原始要点归并）

- IO prompting：`y ~ pθ(y | prompt_IO(x))`，只做输入到输出的直接映射。
- CoT prompting：引入中间 thought 序列 `z`，但通常仍是单链顺序采样。
- CoT-SC：采样多条 CoT 再投票，可提升鲁棒性，但不做局部状态搜索。
- ToT 关注点：在“中间状态层面”引入分支探索与启发式评估。

![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_001.png]]

---

## 4. 论文方法

### 4.1 方法总览

- 方法名称：Tree of Thoughts（ToT）
- 一句话概括方法：把问题求解建模为 thought 树搜索，每个节点是部分解状态。
- 方法整体流程（按步骤写 1/2/3/4）：
  1. 设计 thought 分解粒度（词/句/段/方程步骤）。
  2. 在每个状态生成候选 thought（Sample 或 Propose）。
  3. 评估状态价值（Value 或 Vote）。
  4. 用 BFS/DFS 进行扩展、剪枝、回溯，直到得到最终解。

### 4.2 核心设计

#### 设计 1：Thought Decomposition
- 做了什么：把中间推理过程拆成可搜索的 thought 步。
- 为什么这样设计：便于在“步骤级”做分支探索和状态控制。
- 解决的难点：难点 1。
- 关键实现：不同任务可定义不同 thought 粒度。

#### 设计 2：Thought Generator（Sample / Propose）
- 做了什么：
  - `Sample`：从 CoT 分布独立采样多个候选。
  - `Propose`：同上下文下提出一组互补候选。
- 为什么这样设计：兼顾多样性与去重。
- 解决的难点：难点 1/3。
- 关键公式：`z(i+1) ~ pθ^CoT(z(i+1)|x,z1...zi)`（示意）。

#### 设计 3：State Evaluator（Value / Vote）
- 做了什么：
  - `Value`：独立评估每个状态可行性（如 sure/likely/impossible）。
  - `Vote`：跨状态比较并选择更有前景节点。
- 为什么这样设计：让搜索有可学习启发式，而非盲目扩展。
- 解决的难点：难点 2。

#### 设计 4：Search Algorithm（BFS / DFS）
- 做了什么：
  - BFS：每层保留 top-b 状态，适合深度受限任务。
  - DFS：沿高价值路径深入，失败后回溯。
- 为什么这样设计：根据任务结构和预算做搜索策略匹配。
- 解决的难点：难点 3。

### 4.3 训练 / 推理细节

- 训练阶段做了什么：无需额外训练，主要是推理时框架化调用。
- 推理阶段做了什么：在每步生成候选 thought、评估状态并执行搜索策略。
- 损失函数组成：无新增训练损失（inference-time framework）。
- 关键超参数：树深度、每步候选数 `k`、保留宽度 `b`、评估采样次数。
- 复杂度 / 额外开销：相较单链 CoT 增加调用成本，但显著提升复杂任务成功率。

### 4.4 方法图与流程证据

![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_002.png]]

---

## 5. 论文贡献（只写作者真正新增的东西）

- 贡献 1：提出 ToT，将 IO/CoT/CoT-SC 统一到“思维树搜索”视角。
- 贡献 2：提出可组合模块（分解/生成/评估/搜索），支持任务级适配。
- 贡献 3：在三类差异明显任务上验证 ToT 的有效性与泛化性。

---

## 6. 实验设置

- 数据集 / 任务：
  - Game of 24（算术搜索）
  - Creative Writing（开放式写作规划）
  - Mini Crosswords（自然语言组合搜索）
- 模型：GPT-4（Chat Completion，温度约 0.7）
- 对比方法：IO、CoT、CoT-SC、迭代反思/优化等
- 评价指标：
  - Game of 24：解题成功率
  - Creative Writing：GPT-4 分数 + 人类偏好
  - Mini Crosswords：字母/单词/整题成功率
- 关键设置（原始记录）：
  - Game of 24：BFS，每步保留 `b=5`，值评估采样
  - Creative Writing：深度 2（先计划再写作）
  - Mini Crosswords：DFS，最大搜索步数约 100

### 6.1 实验图证据（任务设置）

![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_003.png]]
![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_006.png]]
![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_008.png]]

---

## 7. 主要结果

### 7.1 主结果

- 结果 1（Game of 24）：ToT 相对 CoT 有显著成功率提升（论文常见对比为 74% vs 4%）。
- 结果 2（Creative Writing）：ToT 的平均连贯性分数与偏好优于 IO/CoT。
- 结果 3（Mini Crosswords）：ToT 大幅提升单词级成功率，并可解出更多完整题目。

### 7.2 从结果中能读出的结论

- 结论 1：关键增益来自“搜索与评估”，而非仅增加样本数。
- 结论 2：不同任务可通过不同搜索策略（BFS/DFS）适配。
- 结论 3：ToT 在需要规划与回溯的任务上优势最明显。

### 7.3 最关键的证据

- 最关键表格：三任务主结果对比表。
- 最关键图：Game of 24 错误分析、Creative Writing 对比、Crosswords 结果图。
- 最关键数字：Game of 24 显著跃升、Creative Writing 人类偏好提升、Crosswords 单词级成功率提升。
- 为什么它最关键：同时覆盖了数学搜索、开放写作和语言谜题三种能力类型。

### 7.4 主结果图（完整）

![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_004.png]]
![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_005.png]]
![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_007.png]]
![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_009.png]]

---

## 8. 消融实验

- 消融点 1：去回溯/弱搜索后性能明显下降（Crosswords 中尤其明显）。
- 消融点 2：状态评估误差会限制上限（oracle 分析显示仍有改进空间）。
- 消融点 3：生成策略（i.i.d. vs propose vs refine）会影响质量-多样性平衡。

---

## 9. 和已有工作的关系

- 这篇论文最接近哪些方法：CoT、CoT-SC、ReAct、RAP、self-reflection、A* 启发式思路。
- 和已有方法相比，最大的不同：ToT 在中间状态层面显式做搜索与评估。
- 真正的新意在哪里：把“语言生成 + 启发式搜索”模块化统一。
- 哪些地方更像“工程改进”而不是“方法创新”：具体 prompt 设计、评估词表和任务模板。
- 这篇论文在整个研究脉络里的位置：从“链式推理”走向“搜索式推理”的代表工作。

---

## 10. 我的理解（这一节不能照抄论文）

### 10.1 直观理解
- 用自己的话解释这篇方法：让模型像下棋一样“想几步、试几条路、再回头改”。
- 它本质上像在做什么：把 LLM 从“单次生成器”变成“可搜索问题求解器”。

### 10.2 我认为最关键的设计
- 最关键设计：状态评估器（value/vote）+ 搜索策略（BFS/DFS）的耦合。
- 为什么我觉得它最关键：这决定了搜索是否真正收敛到更优路径。

### 10.3 我认为最强的一点
- 无需额外训练即可跨任务获得稳定增益。

### 10.4 我认为最可疑的一点
- 调用成本较高，且评估器本身噪声会影响搜索质量。

---

## 11. 局限性

- 局限 1：搜索成本较高，不适合所有已易解任务。
- 局限 2：目前主要验证在三个相对小规模任务上。
- 局限 3：状态评估误判会带来剪枝错误。

### 11.1 讨论与附录证据

- 论文讨论了更广泛影响：更高自主决策能力也可能带来误用风险。
- 论文强调 ToT 可提高可解释性，因为中间 thought 可读。
- 附录给出额外结果与成本分析：主实验成本约百美元量级（按论文记录）。

![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_010.png]]
![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_011.png]]
![[assets/Tree of Thoughts Deliberate Problem Solving with Large Language Models/image_012.png]]

---

## 12. 对我的启发

- 能直接借鉴的部分：先定义中间状态，再定义评估与搜索，不把推理等同于一次解码。
- 不能直接照搬的部分：高调用预算在生产环境需要折中。
- 对我当前课题的启发：可把 ToT 的“分解-评估-回溯”迁移到代码、规划、工具调用任务。
- 可以尝试的改进方向：学习型评估器、检索增强状态表示、自适应搜索深度。
- 可以作为 baseline / 对比项 / ablation 的部分：IO、CoT、CoT-SC、无回溯 ToT。

---

## 13. 待验证问题

- 问题 1：ToT 在代码修复/代理规划这类长期依赖任务上的收益是否同样稳定？
- 问题 2：如何降低 ToT 的调用开销并保持核心增益？
- 问题 3：能否将状态评估器从 prompt 规则升级为学习型模块？

---

## 14. 一句话总结

- ToT 通过“思维分支 + 状态评估 + 搜索回溯”显著提升了 LLM 在复杂推理任务上的求解能力，是从 CoT 迈向搜索式推理的关键工作。

---

## 15. 快速索引（便于二次回看）

- 核心公式：thought 生成分布、状态评估函数、BFS/DFS 搜索策略。
- 核心图表：方法图（image_001/002）、任务结果（image_004/007/009）、附录成本（image_011/012）。
- 最值得复看的章节：Section 3（ToT 框架）与 Section 4（三任务实验）。
- 复现时最需要注意的点：thought 粒度、状态评估提示、搜索预算设置。

### 15.1 整合说明 / 索引

- 原始 Notion 转录内容已拆解进正文 `1~14`。
- 全部 12 张图片已插入到方法/实验/附录对应位置。

### 15.2 导入来源与完整性记录

- 源页面 ID：`b24b752c-a3da-4a1f-9653-2dce8f355133`
- 抓取块数量：`221`
- 未解析块引用：`0`
- 原始 JSON：`notes/_notion_raw/Tree of Thoughts Deliberate Problem Solving with Large Language Models.json`
- 未解析块 ID：无
- 联网校验来源（2026-04-08）：
  - arXiv: https://arxiv.org/abs/2305.10601
  - 代码: https://github.com/princeton-nlp/tree-of-thought-llm

### 15.3 已完成自检记录

- [x] 原始笔记所有内容已整理到模板正文。
- [x] 图片已全部插入并归位。
- [x] 已联网校正论文关键信息并记录来源。


## Wiki 关联

- 参考摘要：[[references/Tree of Thoughts Deliberate Problem Solving with Large Language Models|Tree of Thoughts Deliberate Problem Solving with Large Language Models]]
- 概念锚点：[[concepts/Tree of Thoughts Reasoning]]、[[concepts/Monte Carlo Tree Search for LLM Reasoning]]、[[concepts/Thought Structure Taxonomy]]
- 实体锚点：[[entities/Shunyu Yao]]、[[entities/Dian Yu]]、[[entities/GPT-4]]、[[entities/Di Zhang]]
- 综合页面：[[synthesis/Search-Based Deliberate Reasoning Landscape]]、[[synthesis/Structured Reasoning Methods for LLMs]]、[[synthesis/LLM Reasoning Search and Verification]]
