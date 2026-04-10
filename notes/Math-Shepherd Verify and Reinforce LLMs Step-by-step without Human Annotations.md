---
title: "Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations"
category: note
tags:
  - note
sources:
  - workspace/wiki-update-2026-04-10-global-lint-remediation
created: 2026-04-10
updated: 2026-04-10
summary: "﻿# Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"
---
# Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations

- 阅读日期：[[journal/2026-04-08]]
- 阅读状态：已读
- 标签：#paper #reasoning #process-reward-model #verification #rl
- 相关方向：数学推理、过程监督、奖励建模、LLM 对齐
- 阅读目的：系统梳理“自动过程标注 + PRM 验证/强化学习”这条技术路线

---

## 1. 论文信息

- 题目：Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
- 链接：https://arxiv.org/abs/2312.08935
- 作者：Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, Zhifang Sui
- 单位：Peking University；DeepSeek-AI（论文首页作者单位标注）
- 会议 / 期刊 / 年份：ACL 2024 (Long Paper)
- 关键词（3~8个）：PRM, ORM, Process Supervision, Math Reasoning, Verification, PPO
- 论文一句话主题：在无人工逐步标注下，自动构建过程监督数据训练 PRM，并用于验证和逐步强化学习。

---

## 2. 先看结论（适合快速回顾）

- 这篇论文主要解决什么问题：PRM 效果好但人工逐步标注昂贵，如何自动构建高质量过程监督并有效训练 PRM。
- 提出的核心方法是什么：用“步骤潜力（能否导出正确终答案）”定义步骤质量，借助 completer 采样后续推理，自动打步骤标签（HE/SE）并训练 PRM。
- 最终最重要的结果是什么：逐步 PPO 使 Mistral-7B 从 77.9%→84.1%(GSM8K)、28.6%→33.0%(MATH)；再用 Math-Shepherd 验证可到 89.1%(GSM8K)、43.5%(MATH)。
- 我现在是否值得深入读：值得
- 原因：该工作把“过程奖励模型的监督瓶颈”转成可扩展自动流程，且验证与 RL 两条线都给出正收益。

---

## 3. 问题定义

### 3.1 研究问题
- 论文研究的核心问题：无人工注释地训练可用 PRM，并检验其在验证和强化学习中的收益。
- 输入是什么：数学题 \(p\)、候选推理步骤序列 \(S=(s_1,\dots,s_K)\)、标准答案 \(a^*\)。
- 输出是什么：步骤级奖励 \(r_{s_i}\)，以及用于候选解重排/策略优化的分数。
- 优化目标是什么：提升最终数学题正确率（best-of-N 验证与 greedy 解码 RL 评估）。
- 任务设定 / 威胁模型 / 前提假设：默认可获得题目 gold final answer；步骤好坏由“导向正确终答概率”近似。

### 3.2 为什么重要
- 这个问题为什么值得做：PRM 已被证实优于 ORM，但人工过程标注严重制约规模化。
- 现实应用价值：可用于在线解题重排、离线数据构建、以及 step-level RL 训练。
- 学术上的意义：把 MCTS 风格“潜力估计”引入 LLM 推理监督，建立可扩展过程监督范式。

### 3.3 难点
- 难点 1：步骤正确性难直接定义，且单步标签噪声高。
- 难点 2：多步推理长链路导致末端奖励稀疏，普通 ORM 信号不够细。
- 难点 3：高质量过程监督构造成本高，模型/数据分布错配会影响 PRM 训练。

---

## 4. 论文方法

### 4.1 方法总览
- 方法名称：Math-Shepherd
- 一句话概括方法：用“从中间步骤继续推理后得到正确答案的频率”给步骤打标签，训练 PRM，再用于验证与逐步 PPO。
- 方法整体流程（按步骤写 1/2/3/4）：
  1. 生成器为题目采样多条分步解。
  2. 对每个中间步骤用 completer 采样 N 条后续推理并核对终答。
  3. 以 HE/SE 规则生成步骤标签并训练 PRM。
  4. 用 PRM 做候选重排（verification）或提供 step-level 奖励（RL）。

![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_001.png]]

### 4.2 核心设计
> 每个设计都尽量回答：做了什么、为什么这么设计、解决了哪个难点

#### 设计 1
- 做了什么：定义步骤质量为“潜力（potential）”，即该步骤后续是否/多大概率能到达正确终答。
- 为什么这样设计：避免对中间自然语言步骤做昂贵人工判定。
- 解决的难点：步骤标签定义困难、人工过程标注成本高。
- 关键公式 / 目标函数：
  - \(y_{s_i}^{HE}=1\) 当存在后续采样路径到达 \(a^*\)，否则 0。
  - \(y_{s_i}^{SE}= \frac{1}{N}\sum_{j=1}^N \mathbf{I}(a_j=a^*)\)。
- 证据位置：Method 3.3。

#### 设计 2
- 做了什么：训练 PRM 以步骤级损失监督每一步分数。
- 为什么这样设计：相比 ORM 的序列级单分数，PRM 可提供细粒度反馈。
- 解决的难点：长链路推理中的稀疏奖励与信用分配问题。
- 关键公式 / 目标函数：
  - \(\mathcal{L}_{PRM}=\sum_{i=1}^{K} y_{s_i}\log r_{s_i} + (1-y_{s_i})\log(1-r_{s_i})\)。
- 证据位置：Method 3.2。

#### 设计 3
- 做了什么：验证阶段使用“步骤最小分 + 按终答分组加权”做重排。
- 为什么这样设计：低分步骤通常是解题路径中的致命错误点。
- 解决的难点：best-of-N 中候选多但质量参差，需稳定排序准则。
- 关键公式 / 目标函数：
  - ORM 训练损失：\(\mathcal{L}_{ORM}= y_s\log r_s + (1-y_s)\log(1-r_s)\)。
  - 重排聚合：\(a_{sc+rm}=\arg\max_a\sum_{i=1}^{N}\mathbf{I}(a_i=a)\cdot RM(p,S_i)\)。
- 证据位置：Method 3.2, 3.4。

![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_002.png]]

### 4.3 训练 / 推理细节
- 训练阶段做了什么：
  - 在 MetaMATH 上训练生成器与 completer（3 epochs）。
  - 用 GSM8K/MATH 训练集构造自动过程标签数据；每题采样 15 条解，去重后标注步骤。
  - 奖励模型以 Mistral-7B 为底座，学习率 \(1\times10^{-6}\)，训练 1 epoch。
- 推理阶段做了什么：
  - Verification：每题采样 \(N=256\) 候选解，用 PRM/ORM/SC 重排比较。
  - RL：step-level PPO 在每步结束给奖励，最终用 greedy decode 准确率评估。
- 损失函数组成：ORM 序列级 BCE；PRM 步骤级 BCE（HE/SE 标签）。
- 关键超参数：
  - completer 解码数示例 \(N=8\)（数据构建）；
  - RL 学习率：LLaMA2-7B 为 \(4\times10^{-7}\)，Mistral-7B 为 \(1\times10^{-7}\)；
  - KL 系数 0.04；cosine 调度最小学习率 \(1\times10^{-8}\)；
  - 最大序列长度 512（hfai 3D 并行训练）。
- 复杂度 / 额外开销：步骤级补全显著增算力，但作者强调仍低于人工过程标注成本。

---

## 5. 论文贡献（只写作者真正新增的东西）

- 贡献 1：提出自动过程标注框架，把步骤质量定义为“导向正确终答潜力”，无需人工逐步标注。
- 贡献 2：在验证与逐步强化学习两种设置下系统验证 PRM，覆盖 7B~70B 多模型。
- 贡献 3：给出影响 PRM 的关键因素分析（候选数、completer 强度、预训练底座规模、数据量与 OOD 表现）。

---

## 6. 实验设置

- 数据集：GSM8K、MATH（验证含 MATH500 子集）；OOD 评估用匈牙利国家期末考试（33 题，总分 100）。
- 模型 / 骨干网络：LLaMA2-7B/13B/70B、LLemma-7B/34B、Mistral-7B、DeepSeek-67B。
- 对比方法：Self-Consistency（多数投票）、ORM、RFT、PPO+ORM。
- 评价指标：best-of-N 准确率（验证），greedy decode 准确率（RL）。
- 实现设置：
  - 每题生成 256 候选解用于验证；
  - 训练数据规模约：GSM8K 170k 解、MATH 270k 解（自动构建）。
- 关键超参数：
  - SFT/生成器学习率（示例）：Mistral-7B 为 \(5\times10^{-6}\)，7B/13B 为 \(2\times10^{-5}/1\times10^{-5}\)，34B/67B/70B 为 \(6\times10^{-6}\)。
- 是否开源代码 / 模型：论文与结果公开，官方代码仓库未在论文页明确给出（截至 2026-04-08）。
- 实验是否公平（初步判断）：对比项覆盖充分，但自动标注数据分布与 PRM800K 分布差异会影响横向对比解释。

---

## 7. 主要结果

### 7.1 主结果
- 结果 1：作为 verifier，Math-Shepherd 在 GSM8K/MATH 上整体优于 SC 与 ORM。
- 结果 2：逐步 PPO 明显优于 RFT 与 PPO+ORM，显示步骤级奖励更有效。
- 结果 3：RL 与验证可叠加，组合后进一步提升终准确率。

![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_003.png]]
![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_004.png]]
![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_005.png]]

### 7.2 从结果中能读出的结论
- 结论 1：PRM 在更难的 MATH 上相对 ORM 的优势更明显，细粒度监督对高难推理更关键。
- 结论 2：当奖励模型足够强时，盲目与 SC 叠加不一定继续增益（部分设置会下降）。
- 结论 3：跨分布测试中 PRM 仍显著优于 ORM（匈牙利考试分数高约 9 分）。

![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_011.png]]

### 7.3 最关键的证据
- 最关键表格：验证主结果表（PRM/ORM/SC across models）。
- 最关键图：逐步 PPO 对比图与 OOD 案例图。
- 最关键数字：
  - Mistral-7B：77.9%→84.1%(GSM8K), 28.6%→33.0%(MATH)（step-level PPO）；
  - 验证后：89.1%(GSM8K), 43.5%(MATH)；
  - DeepSeek-67B + verifier：93.3%(GSM8K), 48.1%(MATH)。
- 为什么它最关键：同时覆盖“验证效果、训练效果、泛化效果”三条证据链。

---

## 8. 消融实验

- 消融点 1：
  - 改了什么：候选解数量 \(N\)。
  - 结果如何：PRM 相比 ORM/多数投票的优势随 \(N\) 增大更明显。
  - 说明了什么：PRM 对大候选池的排序能力更强。

![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_006.png]]

- 消融点 2：
  - 改了什么：自动过程标注策略（HE vs SE）与其它标注器（NLI / 规则）。
  - 结果如何：N=4 时 HE 标注准确度约 86%；HE/SE 训练验证器性能接近；均优于 NLI/规则标注。
  - 说明了什么：自动标注质量已足够驱动有效 PRM 训练。

![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_007.png]]
![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_008.png]]

- 消融点 3：
  - 改了什么：奖励模型底座规模与训练数据量。
  - 结果如何：大模型 PRM 更稳健；10k 数据规模下 PRM 约领先 ORM 4%，且上限更高。
  - 说明了什么：PRM 既受模型容量影响，也受数据规模影响，且扩展趋势优于 ORM。

![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_009.png]]
![[assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/image_010.png]]

---

## 9. 和已有工作的关系

- 这篇论文最接近哪些方法：ORM/PRM 奖励建模、SC 验证、RFT/PPO 的 RL 路线。
- 和已有方法相比，最大的不同：不依赖人工逐步标注，改用自动“步骤潜力”标注。
- 真正的新意在哪里：把中间步骤价值估计转化为可扩展训练信号，并在 verifier + RL 双场景闭环验证。
- 哪些地方更像“工程改进”而不是“方法创新”：特定学习率、采样数、并行训练配方。
- 这篇论文在整个研究脉络里的位置：连接“推理验证”与“过程监督 RL”的关键过渡工作。

---

## 10. 我的理解（这一节不能照抄论文）

### 10.1 直观理解
- 用自己的话解释这篇方法：先判断“这一步走下去有没有希望”，再把这个信号用于挑答案和训模型。
- 它本质上像在做什么：把数学推理树的局部节点价值估计，转成可学习的步骤奖励函数。

### 10.2 我认为最关键的设计
- 最关键设计：用 completer 采样后续轨迹来定义步骤潜力（HE/SE）。
- 为什么我觉得它最关键：它同时解决“标签从哪来”和“标签如何与最终正确率对齐”两个核心问题。

### 10.3 我认为最强的一点
- 在无人工步骤标注条件下，仍能稳定获得比 ORM 更高的验证效果与 RL 提升。

### 10.4 我认为最可疑的一点
- 对 completer 质量和采样成本较敏感，数据构建成本仍可能成为大规模训练瓶颈。

---

## 11. 局限性

- 局限 1：自动标注要对每个步骤做多次后续补全，计算成本高。
- 局限 2：自动标签含噪，且与人工 PRM800K 的分布不一致会影响比较解释。
- 局限 3：方法依赖可核验 final answer 的任务形态，迁移到开放式任务需要新判分机制。

---

## 12. 对我的启发

- 能直接借鉴的部分：把“终答案正确性”回传到中间步骤，用作过程监督。
- 不能直接照搬的部分：固定 HE/SE 规则在开放域推理可能不稳，需要任务化改造。
- 对我当前课题的启发：可把复杂任务拆为步骤级价值学习，再做重排/策略优化。
- 可以尝试的改进方向：更便宜的步骤价值估计（缓存、早停、蒸馏、轻量 verifier）。
- 可以作为 baseline / 对比项 / ablation 的部分：SC、ORM、RFT、PPO+ORM、PRM(HE/SE)。

---

## 13. 待验证问题

- [ ] 问题 1：若将 PRM 分数聚合从“最小步分”改为“软最小值/风险敏感聚合”，验证效果是否更稳？
- [ ] 问题 2：在编码、规划等非数学任务中，步骤潜力标签如何定义才可自动化且低噪？
- [ ] 问题 3：能否用小 completer + 主动采样策略逼近大 completer 的标注质量？

---

## 14. 一句话总结

- Math-Shepherd 证明了“自动过程监督”可以实用化 PRM，并在数学验证与逐步 RL 中同时带来稳定增益。

---

## 15. 快速索引（便于二次回看）

- 核心公式：\(L_{ORM}\)、\(L_{PRM}\)、\(y_{s_i}^{HE}\)、\(y_{s_i}^{SE}\)、\(a_{sc+rm}\)。
- 核心图表：image_002（方法流程）、image_003/004/005（主结果）、image_006~010（分析）、image_011（OOD）。
- 最值得复看的章节：3.1~3.3、4.2、7.1、8。
- 复现时最需要注意的点：completer 质量、候选采样数 N、PRM 聚合规则、step-level PPO 超参数。

### 15.1 整合说明 / 索引

- 原始转录内容已完整拆入 1~14 节正文。
- 本节仅保留二次回看索引，不保留原始堆放内容。

### 15.2 导入来源与完整性记录

- 论文来源（联网校验日期：2026-04-08）：
  - arXiv：https://arxiv.org/abs/2312.08935
  - ACL Anthology：https://aclanthology.org/2024.acl-long.510/
- 本地资源：
  - 原始 JSON：`notes/_notion_raw/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations.json`
  - 图片目录：`notes/assets/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations/`
- 完整性：
  - 原笔记方法/实验/分析/局限/结论信息均已并入模板正文。
  - 图片 `image_001` 至 `image_011` 已按语义位置插入正文。


## Wiki 关联

- 参考摘要：[[references/Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations|Math-Shepherd Verify and Reinforce LLMs Step-by-step without Human Annotations]]
- 概念锚点：[[concepts/Process-Supervised Step Verification]]、[[concepts/Implicit Chain-of-Thought Internalization]]、[[concepts/Tree of Thoughts Reasoning]]
- 实体锚点：[[entities/Peiyi Wang]]、[[entities/Lei Li]]、[[entities/GPT-4]]
- 综合页面：[[synthesis/Process Supervision and CoT Internalization]]、[[synthesis/Structured Reasoning Methods for LLMs]]、[[synthesis/LLM Reasoning Search and Verification]]
