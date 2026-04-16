---
title: "From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step"
category: note
tags:
  - note
sources:
  - workspace/wiki-update-2026-04-10-global-lint-remediation
created: 2026-04-10
updated: 2026-04-10
summary: "﻿# From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step"
---
# From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step

- 阅读日期：[[journal/2026-04-08]]
- 阅读状态：已读
- 标签：#paper #imported #notion
- 相关方向：LLM 推理、CoT 内化、课程学习
- 阅读目的：梳理从显式 CoT 到隐式 CoT 的训练路径

---

## 1. 论文信息

- 题目：From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step
- 链接：https://arxiv.org/abs/2405.14838
- 作者：Yuntian Deng, Yejin Choi, Stuart Shieber
- 单位：待逐一补全（已核对作者与 arXiv 版本）
- 发表：arXiv 2024（ICLR 2025 提交稿）
- 关键词：Implicit CoT, Stepwise Internalization, curriculum learning, removal smoothing, reasoning efficiency

---

## 2. 先看结论（适合快速回顾）

- 这篇论文主要解决什么问题：显式 CoT 准确但慢，隐式推理快但训练难。
- 提出的核心方法是什么：Stepwise Internalization（SI），逐步移除 CoT token 并持续微调。
- 最终最重要的结果是什么：在保持准确率的同时显著提高推理速度；GPT-2 small 可达高精度乘法，Mistral-7B 在 GSM8K 无中间步骤仍可超过 50%。
- 我现在是否值得深入读：值得
- 原因：对“高性能+低延迟”推理系统具有直接工程价值。

---

## 3. 问题定义

### 3.1 研究问题

- 论文关注的核心问题：如何把“训练时依赖显式 CoT”转化为“推理时不输出中间步骤”的隐式推理能力。
- 为什么这个问题重要：显式 CoT token 长、成本高，且在长链路下延迟明显增加。
- 论文要优化或解决的目标：让模型逐步内化中间推理步骤，在减少/去除 CoT 输出后尽可能保持任务准确率。

### 3.2 为什么重要

- 这个问题为什么值得做：显式 CoT 的 token 成本会放大训练与推理开销。
- 现实应用价值：提升延迟敏感场景中的推理质量/速度平衡。
- 学术上的意义：探索“监督步骤”如何转化为“内部能力”。

### 3.3 难点

- 难点 1：直接移除 CoT token 会导致损失函数分布突变，训练不稳定。
- 难点 2：优化器（如 AdamW）二阶统计会在突变时失效，需要状态重置策略。
- 难点 3：若移除调度过于激进，模型无法跟上目标变化，容易崩溃到低精度。

![[notes/assets/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step/image_001.png]]

---

## 4. 论文方法

### 4.1 方法总览

- 方法名称：Stepwise Internalization（SI）/ ICoT-SI
- 整体思路：从显式 CoT 模型出发，按课程学习调度逐步减少中间步骤 token，使推理过程内化到模型隐藏状态。
- 代理目标 / 损失 / 优化目标：逐步改变训练目标，从 `P(y, z|x)` 过渡到仅 `P(y|x)`，并保持任务精度。

### 4.2 核心设计

#### 设计 1：逐步移除 CoT token 的课程调度
- 做了什么：定义移除函数 `s(t)`，随训练步数增加逐步移除更多 CoT token。
- 为什么这么设计：避免一次性去除导致训练断崖。
- 关键实现：
  - 初始目标：`min -log Pθ(y, z_{1:m}|x)`
  - 逐步目标：`min -log Pθ(y, z_{1+s(t):m}|x)`
  - 线性调度：`s(t)= floor(Δ * t / T)`

#### 设计 2：Removal Smoothing
- 做了什么：在移除数量上加随机偏移 `o`，形成平滑移除 `s*(t)=s(t)+o`。
- 为什么这么设计：缓解相邻阶段损失函数的跳变。
- 关键实现：`P(o) ∝ exp(-λo)`，大多数时间不额外移除，少量时间提前移除更多 token。

#### 设计 3：优化器状态重置与稳定训练
- 做了什么：每次增加移除量时重置优化器状态。
- 为什么这么设计：避免 AdamW 对历史二阶统计的错误累积。
- 关键实现：与 removal smoothing 配合可显著降低“训练崩溃”概率。

![[notes/assets/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step/image_002.png]]

### 4.3 训练 / 推理细节

- 训练阶段做了什么：先显式 CoT 训练，再执行 SI 课程式移除调度并持续微调。
- 推理阶段做了什么：减少/去除中间步骤，直接生成最终答案。
- 损失函数组成：语言建模损失，随移除调度动态改变监督目标。
- 关键超参数：`Δ`（每 epoch 移除速率）、`λ`（平滑分布）、优化器重置时机。
- 复杂度 / 额外开销：训练更复杂，但推理显著更快。

---

## 5. 论文贡献（只写作者真正新增的东西）

- 贡献 1：提出 SI（Stepwise Internalization）训练框架，实现显式 CoT 到隐式 CoT 的平滑过渡。
- 贡献 2：提出并验证 removal smoothing + optimizer reset 的稳定训练组合。
- 贡献 3：在乘法与 GSM8K 上展示“高速度 + 可接受准确率”的实证折中。

---

## 6. 实验设置

- 数据集：
  - 多位数乘法（4×4、5×5、7×7、9×9 及更大设置）
  - GSM8K（含增强数据）
- 模型：GPT-2 Small、Phi-3 3.8B、Mistral-7B
- 对比方法：No-CoT、Explicit-CoT、ICoT-KD
- 评价指标：准确率、推理速度（H100，batch=1 下 samples/s）
- 关键超参数：
  - AdamW，学习率约 `5e-5 ~ 1e-4`
  - 批大小与梯度累积按模型规模调整
  - 单卡 H100（80GB）训练，最长约 24h

![[notes/assets/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step/image_003.png]]

---

## 7. 主要结果

### 7.1 主结果

- 结果 1：ICoT-SI 使 GPT-2 Small 在 9×9 乘法达到约 0.99 准确率。
- 结果 2：在 GSM8K 上，Mistral-7B 无中间步骤仍可超过 0.50 准确率。
- 结果 3：相较显式 CoT，ICoT-SI 准确率略低但速度显著更快。

### 7.2 从结果中能读出的结论

- 结论 1：逐步内化能显著降低对显式步骤输出的依赖。
- 结论 2：Removal smoothing 与优化器重置对稳定训练关键。
- 结论 3：可在准确率与速度间形成可调折中。

### 7.3 最关键的证据

- 最关键表格：乘法与 GSM8K 主结果对比表。
- 最关键图：SI 训练流程图、准确率-速度曲线、消融曲线。
- 最关键数字：9×9 乘法约 0.99；GSM8K 无中间步骤 >0.50。
- 为什么它最关键：直接验证“内化能力”而非仅格式改变。

![[notes/assets/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step/image_004.png]]
![[notes/assets/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step/image_005.png]]
![[notes/assets/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step/image_006.png]]

---

## 8. 消融实验

- 消融点 1：
  - 改了什么：移除 removal smoothing。
  - 结果如何：训练更容易出现精度崩溃。
  - 说明了什么：平滑移除是稳定性的关键机制。

- 消融点 2：
  - 改了什么：不做 optimizer reset。
  - 结果如何：在移除阶段切换时不稳定。
  - 说明了什么：二阶统计历史会放大分布突变问题。

- 消融点 3：
  - 改了什么：更激进的每轮移除 token 数（更大 `Δ`）。
  - 结果如何：训练更快但更易失败。
  - 说明了什么：移除速度存在稳定性上限。

![[notes/assets/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step/image_007.png]]
![[notes/assets/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step/image_008.png]]

---

## 9. 和已有工作的关系

- 这篇论文最接近哪些方法：CoT、ICoT-KD、课程学习、知识蒸馏。
- 和已有方法相比，最大的不同：显式步骤逐步移除而非一次性蒸馏。
- 真正的新意在哪里：将中间步骤作为训练时可退火监督。
- 哪些地方更像“工程改进”而不是“方法创新”：调度函数与优化稳定策略。
- 这篇论文在整个研究脉络里的位置：隐式推理训练方法的重要补充。

---

## 10. 我的理解（这一节不能照抄论文）

- 直观理解：先让模型“写出推理过程”，再逐步“收回外显步骤”，把能力沉到参数里。
- 最值得关注的设计：remove smoothing + optimizer reset 的组合，不是简单调参，而是应对目标分布突变的核心机制。
- 和已有方法相比的新意：不是纯蒸馏，而是课程式内化，能提供更连续的准确率-速度折中曲线。
- 我认为最强的一点：在不输出中间步骤时仍能保持可用准确率。

---

## 11. 局限性

- 局限 1：训练成本较高，每次移除阶段都需要持续微调。
- 局限 2：参数（`Δ`、`λ`）敏感，激进设置会导致不稳定。
- 局限 3：隐式推理可解释性下降，失去显式 CoT 的可审计中间链路。

---

## 12. 对我的启发

- 能直接借鉴的部分：逐步退火监督信号的训练范式。
- 不能直接照搬的部分：调度参数对任务/模型规模敏感。
- 对我当前课题的启发：可把“先显式后隐式”用于多步规划训练。
- 可以尝试的改进方向：自适应调度、在线难度估计、混合显隐式解码。
- 可以作为 baseline / 对比项 / ablation 的部分：无 CoT、显式 CoT、ICoT-KD。

---

## 13. 待验证问题

- 问题 1：SI 在代码推理、规划任务上是否同样稳定有效？
- 问题 2：是否可以自动学习每层/每样本的移除节奏而非固定调度？
- 问题 3：能否通过探针方法恢复隐式推理内部状态，提高可解释性？

---

## 14. 一句话总结

- SI 通过“课程式移除 CoT token + 稳定训练策略”把显式推理逐步内化为隐式能力，在速度与准确率之间提供了实用折中。

---

## 15. 快速索引（便于二次回看）

- 核心公式：`s(t)= floor(Δ * t/T)`、`s*(t)=s(t)+o`、`P(o)∝exp(-λo)`。
- 核心图表：机制图（image_001/002）、主结果（image_004/005）、速度精度权衡（image_006）、消融（image_007/008）。
- 最值得复看的章节：方法章节与消融章节。
- 复现时最需要注意的点：优化器重置、调度强度与稳定性。

### 15.1 整合说明 / 索引

- 原始 Notion 完整转录内容已拆分并归位到 `1~14`。
- 全部图片已按方法/实验语义插入正文位置。

### 15.2 导入来源与完整性记录

- 源页面 ID：`dcbf57f0-4c05-4a08-8f26-e67df60bda7f`
- 抓取块数量：`125`
- 未解析块引用：`0`
- 原始 JSON：`notes/_notion_raw/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step.json`
- 未解析块 ID：无
- 联网校验来源（2026-04-08）：
  - arXiv: https://arxiv.org/abs/2405.14838
  - 代码仓库: https://github.com/da03/Internalize_CoT_Step_by_Step

### 15.3 已完成自检记录

- [x] 原始笔记所有内容已并入模板正文。
- [x] 图片均已插入并保留在对应位置。
- [x] 已联网检索并补充核心论文信息。


## Wiki 关联

- 参考摘要：[[references/From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step|From Explicit CoT to Implicit CoT Learning to Internalize CoT Step by Step]]
- 概念锚点：[[concepts/Implicit Chain-of-Thought Internalization]]、[[concepts/Removal Smoothing for CoT Internalization]]、[[concepts/Process-Supervised Step Verification]]
- 实体锚点：[[entities/Yuntian Deng]]、[[entities/Yejin Choi]]、[[entities/Stuart Shieber]]
- 综合页面：[[synthesis/Process Supervision and CoT Internalization]]、[[synthesis/Structured Reasoning Methods for LLMs]]、[[synthesis/LLM Reasoning Search and Verification]]

