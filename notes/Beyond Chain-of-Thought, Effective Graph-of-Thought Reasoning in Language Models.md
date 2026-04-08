# Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models

- 阅读日期：2026-04-08
- 阅读状态：已读
- 标签：#paper #imported #notion
- 相关方向：推理增强、图结构表示、多模态推理、CoT 扩展
- 阅读目的：评估“图式思维表示”相对线性 CoT 的真实增益及可复用模块

---

## 1. 论文信息

- 题目：Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models
- 链接：https://arxiv.org/abs/2305.16582
- 作者：Yao Yao, Zuchao Li, Hai Zhao
- 单位：Shanghai Jiao Tong University；Wuhan University
- 会议 / 期刊 / 年份：arXiv / 2023（更新版 2024）
- 关键词（3~8个）：Graph-of-Thought, ECC, GAT, gated fusion, multimodal reasoning, AQUA-RAT, ScienceQA
- 论文一句话主题：将线性 CoT 扩展为图结构推理表示，以提升文本与多模态复杂推理性能。

---

## 2. 先看结论（适合快速回顾）

- 这篇论文主要解决什么问题：线性 CoT 难以建模人类非顺序、跳跃式思维，导致复杂推理关系利用不足。
- 提出的核心方法是什么：GoT（Graph-of-Thought）两阶段框架：ECC 构图 + GAT 编码 + 门控融合。
- 最终最重要的结果是什么：AQUA-RAT 与 ScienceQA 均取得提升；ScienceQA 从 85.19% 提升到 87.59%。
- 我现在是否值得深入读：值得
- 原因：思维图构建与融合模块可迁移到其他推理管线，且有完整消融支持。

---

## 3. 问题定义

### 3.1 研究问题
- 论文研究的核心问题：如何把非线性思维关系注入 LLM 推理过程。
- 输入是什么：问题文本（及可选图像）+ 由输入构建的思维图。
- 输出是什么：阶段1生成理由，阶段2生成最终答案。
- 优化目标是什么：提高理由质量与最终答案准确率。
- 任务设定 / 威胁模型 / 前提假设：假设 OpenIE + 共指可构建有效图结构；两阶段推理可分离建模。

### 3.2 为什么重要
- 这个问题为什么值得做：很多推理需要跨句跨实体关系，线性链式中容易丢失结构信息。
- 现实应用价值：提升复杂问答与多模态教育类问题准确率。
- 学术上的意义：从“序列化推理”迈向“结构化推理表示”。

### 3.3 难点
- 难点 1：自然语言到图结构的可靠映射难。
- 难点 2：文本-图像-思维图三模态融合易引入噪声。
- 难点 3：需证明改进源于图结构而非参数量或训练技巧。

---

## 4. 论文方法

### 4.1 方法总览
- 方法名称：Graph-of-Thought (GoT) Reasoning
- 一句话概括方法：先构建思维图，再通过图编码和门控融合增强两阶段推理。
- 方法整体流程（按步骤写 1/2/3/4）：
  1. 使用 ECC 从输入抽取并构建思维图。
  2. 分别编码文本、图像（可选）和思维图特征。
  3. 通过注意力对齐与门控融合形成联合表示。
  4. 阶段1生成理由，阶段2基于理由+输入生成答案。

### 4.2 核心设计
> 每个设计都尽量回答：做了什么、为什么这么设计、解决了哪个难点

#### 设计 1
- 做了什么：ECC（Extract-Cluster-Coreference）构图流程。
- 为什么这样设计：将离散跳跃思维关系显式结构化。
- 解决的难点：难点 1（从文本到结构）。
- 关键公式 / 目标函数：以 OpenIE 三元组和共指消解构建节点/边与邻接矩阵。
- 证据位置：Section 2.1；图见 `image_003`、`image_004`。

#### 设计 2
- 做了什么：GoT 编码器（节点嵌入 + GAT 多头注意力）。
- 为什么这样设计：学习节点间关系并提取图结构语义。
- 解决的难点：难点 2（结构信息有效表达）。
- 关键公式 / 目标函数：GAT 注意力权重与节点更新公式（见正文公式组）。
- 证据位置：Section 2.2；图见 `image_005`。

#### 设计 3
- 做了什么：文本/图像/思维图的注意力对齐与门控融合。
- 为什么这样设计：统一多模态证据，抑制无关噪声。
- 解决的难点：难点 2/3（融合有效性与真实性能来源）。
- 关键公式 / 目标函数：门控系数 `lambda` 与融合输出 `H` 公式。
- 证据位置：Section 2.3；公式与主结果/消融对应。

### 4.3 训练 / 推理细节
- 训练阶段做了什么：对两阶段框架进行监督微调（理由生成 + 答案生成）。
- 推理阶段做了什么：先生成理由，再结合理由进行答案推断。
- 损失函数组成：序列生成目标（以论文设定为准）。
- 关键超参数：学习率 `5e-5`，训练 `100 epochs`，4×A800 80G。
- 复杂度 / 额外开销：增加构图与图编码成本，训练略慢于普通 CoT。

### 4.4 方法流程图与构图证据
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_001.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_002.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_003.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_004.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_005.png]]

---

## 5. 论文贡献（只写作者真正新增的东西）

- 贡献 1：提出 GoT，将推理表示从链扩展为图。
- 贡献 2：提出 ECC + GoT 编码 + 门控融合的完整两阶段多模态框架。
- 贡献 3：在 AQUA-RAT 与 ScienceQA 上均超越强 CoT 基线并有消融支撑。

> 判断标准：如果删掉这一点，论文是否还成立？如果“是”，那它可能不是核心贡献。

---

## 6. 实验设置

- 数据集：AQUA-RAT、ScienceQA。
- 模型 / 骨干网络：FLAN-Alpaca (T5-base/T5-large)，视觉编码器 ViT-large（部分实验 UnifiedQA+DETR）。
- 对比方法：Zero-shot CoT、Few-shot/Manual-CoT、Auto-CoT、Multimodal-CoT、两阶段 CoT 微调等。
- 评价指标：Accuracy、ROUGE-L，并做学科/年级分层分析。
- 实现设置：4×A800 80G，100 epochs，lr=5e-5。
- 关键超参数：构图策略、融合模块设置、骨干规模。
- 是否开源代码 / 模型：是（GitHub）。
- 实验是否公平（初步判断）：整体较充分，且含“参数扩大但非 GoT”的对照。

### 6.1 数据与训练附录图
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_014.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_015.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_016.png]]

---

## 7. 主要结果

### 7.1 主结果
- 结果 1：AQUA-RAT 准确率 30.09% -> 32.09%（GoT）。
- 结果 2：ScienceQA 准确率 85.19% -> 87.59%（相对 Multimodal-CoT +2.40%）。
- 结果 3：理由生成阶段 ROUGE-L 也有提升（ScienceQA 约 +1.15）。

### 7.2 从结果中能读出的结论
- 结论 1：结构化思维图对答案阶段帮助更明显。
- 结论 2：在小参数模型上，GoT 可显著缩小与大模型提示式方法差距。
- 结论 3：多模态场景中图结构信息不是“可有可无”。

### 7.3 最关键的证据
- 最关键表格：AQUA-RAT 与 ScienceQA 主结果表（见 `image_006`、`image_007`）。
- 最关键图：消融结果图（`image_009`、`image_010`）。
- 最关键数字：32.09%、87.59%、+2.40%、-1.78%（随机图消融）。
- 为什么它最关键：同时证明“有效果”和“不是参数幻觉”。

### 7.4 主结果与分析图证据
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_006.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_007.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_008.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_009.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_010.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_011.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_012.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_013.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_017.png]]

---

## 8. 消融实验

- 消融点 1：
  - 改了什么：Random Thought Graph。
  - 结果如何：AQUA-RAT 下降约 1.78%。
  - 说明了什么：结构正确性本身是关键，随机图无效甚至有害。

- 消融点 2：
  - 改了什么：Triplets Concatenation（仅拼接三元组，不做图化）。
  - 结果如何：准确率约 31.20%，低于完整 GoT。
  - 说明了什么：图结构优于平铺三元组文本。

- 消融点 3：
  - 改了什么：Coreference Injection（直接替换文本共指）。
  - 结果如何：约 -1.7%。
  - 说明了什么：粗暴替换会引入噪声，结构化表示更稳健。

---

## 9. 和已有工作的关系

- 这篇论文最接近哪些方法：CoT prompting、CoT 微调、Multimodal-CoT。
- 和已有方法相比，最大的不同：把推理单元显式建图并与多模态联合编码。
- 真正的新意在哪里：ECC 构图 + GAT 编码 + 两阶段门控融合的完整联动。
- 哪些地方更像“工程改进”而不是“方法创新”：具体编码器替换和训练配方细节。
- 这篇论文在整个研究脉络里的位置：从“链式推理”向“结构化推理表示”过渡的代表工作。

---

## 10. 我的理解（这一节不能照抄论文）

### 10.1 直观理解
- 用自己的话解释这篇方法：先把问题拆成“可连边的思维节点图”，再让模型沿图关系推理。
- 它本质上像在做什么：给 CoT 增加一层可学习的“关系骨架”。

### 10.2 我认为最关键的设计
- 最关键设计：ECC 质量与图编码质量的耦合。
- 为什么我觉得它最关键：图构坏了，后续融合再强也难补救。

### 10.3 我认为最强的一点
- 在多模态任务上依然稳定提升，证明该思想不局限于纯文本链式场景。

### 10.4 我认为最可疑的一点
- 对 OpenIE / 共指工具链依赖强，跨领域迁移时可能脆弱。

### 10.5 案例分析补充图
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_018.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_019.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_020.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_021.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_022.png]]
![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_023.png]]

---

## 11. 局限性

- 局限 1：额外计算开销与训练时延。
- 局限 2：构图质量受外部 NLP 组件上限限制。
- 局限 3：理由生成阶段提升有限，收益更多体现在答案阶段。

![[assets/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models/image_024.png]]

> 可从假设过强、实验覆盖不足、开销过大、泛化不明、复现风险高等角度写。

---

## 12. 对我的启发

- 能直接借鉴的部分：构图-编码-融合三段式模块化设计。
- 不能直接照搬的部分：OpenIE/共指工具链在中文或领域文本上的可靠性。
- 对我当前课题的启发：可以把“结构先验”加入复杂推理任务而不仅依赖提示词。
- 可以尝试的改进方向：可学习构图、检索增强图节点、图稀疏化。
- 可以作为 baseline / 对比项 / ablation 的部分：随机图、三元组拼接、共指直接注入。

---

## 13. 待验证问题

- [ ] 问题 1：端到端可学习构图能否稳定替代 OpenIE + CoreNLP？
- [ ] 问题 2：GoT 在编程/规划任务上是否仍优于线性 CoT？
- [ ] 问题 3：在更长上下文下图规模如何控制而不损失性能？

---

## 14. 一句话总结

- GoT 通过把“思维链”升级为“思维图”，在文本和多模态推理任务上都带来了稳定且有证据支撑的性能提升。

---

## 15. 快速索引（便于二次回看）

- 核心公式：GAT 注意力、门控融合 `lambda` / `H`。
- 核心图表：`image_006`、`image_007`（主结果）；`image_009`、`image_010`（消融）。
- 最值得复看的章节：Section 2.1~2.3（构图与融合）+ Section 4.2（消融）。
- 复现时最需要注意的点：构图质量、共指处理、融合模块稳定性。

### 15.1 整合说明 / 索引

- 原始 Notion 转录内容已拆入正文各节：方法在 `4`，实验在 `6-8`，分析与局限在 `10-11`。
- 全部图片已按语义位置转移至正文，不再在 `15.1` 堆放原文。
- 来源与导入信息统一保留在 `15.2`。

### 15.2 导入来源与完整性记录

- 源页面 ID：`d589b5e4-757a-4bec-a6fd-518e6df7b3e6`
- 抓取块数量：`197`
- 未解析块引用：`0`
- 原始 JSON：`notes/_notion_raw/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models.json`
- 未解析块 ID：

- 联网校验来源（2026-04-08）：
  - arXiv: https://arxiv.org/abs/2305.16582
  - GitHub: https://github.com/Zoeyao27/Graph-of-Thought



## Wiki 关联

- 参考摘要：[[references/Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models|Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Language Models]]
- 概念锚点：[[concepts/Graph of Thoughts Reasoning]]、[[concepts/Thought Structure Taxonomy]]、[[concepts/Tree of Thoughts Reasoning]]
- 实体锚点：[[entities/Yao Yao]]、[[entities/Hai Zhao]]、[[entities/GPT-4]]
- 综合页面：[[synthesis/Search-Based Deliberate Reasoning Landscape]]、[[synthesis/Structured Reasoning Methods for LLMs]]、[[synthesis/LLM Reasoning Search and Verification]]
