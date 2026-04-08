# Maintaining Plasticity in Deep Continual Learning

- 阅读日期：2026-04-08
- 阅读状态：已读
- 标签：#paper #imported #notion #continual-learning #plasticity
- 相关方向：持续学习、优化与表征退化、深度强化学习
- 阅读目的：系统整理“可塑性丧失”问题、理解 Continual Backpropagation（CBP）算法、提炼可复现实验设置

---

## 1. 论文信息

- 题目：Maintaining Plasticity in Deep Continual Learning（后续正式发表标题：Loss of plasticity in deep continual learning）
- 链接：
  - arXiv（预印本）：https://arxiv.org/abs/2306.13812
  - Nature（正式版）：https://www.nature.com/articles/s41586-024-07711-7
  - 代码：https://github.com/shibhansh/loss-of-plasticity
- 作者：
  - 预印本（arXiv 源文件）：Shibhansh Dohare, J. Fernando Hernandez-Garcia, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton
  - 正式版（Nature 页面）：在上述基础上包含 Qingfeng Lan
- 单位：
  - Department of Computing Science, University of Alberta
  - Canada CIFAR AI Chair / Alberta Machine Intelligence Institute (Amii)
- 会议 / 期刊 / 年份：
  - arXiv：2023（2306.13812）
  - Nature：632, 768–774 (2024)，DOI: 10.1038/s41586-024-07711-7
- 关键词（3~8个）：loss of plasticity, continual learning, continual ImageNet, online permuted MNIST, effective rank, continual backpropagation, shrink-and-perturb, L2 regularization
- 论文一句话主题：标准深度学习在持续学习中会逐步丧失学习新任务能力，而“持续注入随机性”的方法（尤其 CBP）可显著维持可塑性。

![[assets/Maintaining Plasticity in Deep Continual Learning/image_001.png]]

---

## 2. 先看结论（适合快速回顾）

- 这篇论文主要解决什么问题：不是“遗忘旧任务”，而是更基础的“对新任务逐步学不动”（loss of plasticity）。
- 提出的核心方法是什么：Continual Backpropagation（CBP）在每次更新后按小比例替换低效隐藏单元，输入权重重采样、输出权重置零，并用成熟阈值避免刚替换单元被立刻再替换。
- 最终最重要的结果是什么：
  - 标准反向传播在 Continual ImageNet 上前期上升、后期显著下滑，最终接近或低于线性基线。
  - L2 正则与 Shrink-and-Perturb（S&P）可缓解，但超参数敏感。
  - CBP 在较宽超参数范围内更稳定，且在长任务序列（图中到 5000 任务）保持更高表现。
- 我现在是否值得深入读：值得
- 原因：给出“可塑性丧失”的系统证据链、可观测机理指标和可实施算法，实验覆盖监督学习与强化学习场景。

---

## 3. 问题定义

### 3.1 研究问题
- 论文研究的核心问题：深度网络在持续学习时，是否会在长时程训练中丧失继续学习新任务的能力。
- 输入是什么：
  - 持续到来的任务流（Continual ImageNet、Online Permuted MNIST、Slowly-Changing Regression）。
  - 每个任务的新样本序列。
- 输出是什么：
  - 每个任务结束时的测试性能（分类准确率或回归误差）。
  - 解释可塑性变化的结构性指标（死单元比例、权重规模、有效秩）。
- 优化目标是什么：
  - 表层目标：持续提升/维持新任务学习能力。
  - 方法目标：减缓或消除可塑性随任务推进而衰减。
- 任务设定 / 威胁模型 / 前提假设：
  - 任务持续变化但难度基本稳定（尤其 Continual ImageNet 通过类别对构造）。
  - 不依赖回放大缓冲将问题伪装为 train-once。
  - 重点区分“可塑性丧失”和“灾难性遗忘”。

### 3.2 为什么重要
- 这个问题为什么值得做：
  - 深度学习主流流程偏“训练一次再部署”；真实系统常需长期在线适应。
  - 若可塑性退化，即使不考虑遗忘，也会出现“新任务学不动”。
- 现实应用价值：
  - 机器人、在线推荐、流式感知、强化学习智能体都需要长期持续更新。
  - 仅重训解决会造成计算和部署成本不可接受。
- 学术上的意义：
  - 把持续学习研究从“保留旧知识（stability）”扩展到“维持新学习能力（plasticity）”。
  - 提供可验证的机理指标与算法干预路径。

### 3.3 难点
- 难点 1：可塑性不是单一标量，需用多指标联合刻画（死单元、权重幅度、有效秩）。
- 难点 2：很多一次性训练里有效的方法（Adam、Dropout、归一化）在持续设置下可能反向作用。
- 难点 3：既要持续注入多样性，又要避免破坏已学功能，替换策略需“选择性且稳定”。

---

## 4. 论文方法

### 4.1 方法总览
- 方法名称：Continual Backpropagation（CBP）
- 一句话概括方法：在常规 SGD/反向传播基础上，持续替换少量“低效用”隐藏单元以维持网络多样性与可塑性。
- 方法整体流程（按步骤写 1/2/3/4）：
  1. 常规前向与反向传播更新权重。
  2. 维护每层每个隐藏单元的年龄与效用统计。
  3. 在可替换集合（年龄超过成熟阈值）里选效用最低单元。
  4. 重置其输入权重为初始化采样、输出权重置零，并重置该单元统计量。

### 4.2 核心设计
> 每个设计都尽量回答：做了什么、为什么这么设计、解决了哪个难点

#### 设计 1：效用驱动的选择性重初始化
- 做了什么：按“整体效用”而非随机挑选单元替换，替换率由 `ρ` 控制。
- 为什么这样设计：在“注入随机性”与“保留已学能力”之间折中，优先替换贡献低、适应弱的单元。
- 解决的难点：难点 3。
- 关键公式 / 目标函数：
  - 有效秩定义（论文用于分析表征退化）：
```latex
\begin{align}
\text{erank}(\Phi) \overset{.}{=} \exp\left\{ H(p_1, p_2, ..., p_q)  \right\}, \text{where } 
H(p_1, p_2, ..., p_q) = - \sum^q_{k=1} p_k \log(p_k).
\end{align}
```
- 证据位置：正文 “Understanding Loss of Plasticity” + 公式块（Notion 原始公式）。

#### 设计 2：贡献效用 + 适应效用的联合
- 做了什么：将单元效用定义为贡献效用与适应效用结合，并用滑动平均与偏置修正稳定估计。
- 为什么这样设计：单看输出贡献会忽略“未来可调整性”，单看权重幅度又无法反映当前贡献。
- 解决的难点：难点 1、难点 3。
- 关键公式 / 目标函数：
```latex
\begin{align}
c_{l, i, t} &= \eta*c_{l, i, t-1} + (1 - \eta)* |h_{l, i, t}| *\sum_{k=1}^{n_{l+1}} |w_{l, i, k, t}|,
\end{align}
```

```latex
\begin{align}
f_{l, i, t} &= \eta*f_{l, i, t-1} + (1 - \eta)* h_{l, i, t}, \\
\hat{f}_{l, i, t} &= \frac{f_{l, i, t-1}}{1 - \eta^{a_{l, i, t}}}, \\
z_{l, i, t} &= \eta*z_{l, i, t-1} + (1 - \eta)* |h_{l, i, t} - \hat{f}_{l, i, t}| *\sum_{k=1}^{n_{l+1}} |w_{l, i, k, t}|,
\end{align}
```

```latex
\begin{align}
y_{l, i, t} &= \frac{|h_{l, i, t} - \hat{f}_{l, i, t}| *\sum_{k=1}^{n_{l+1}} |w_{l, i, k, t}|}{\sum_{j=1}^{n_{l-1}} |w_{l-1, j, i, t}|} \\
u_{l, i, t} &= \eta* u_{l, i, t-1} + (1 - \eta)* y_{l, i, t}, \\
\hat{u}_{l, i, t} &= \frac{u_{l, i, t-1}}{1 - \eta^{a_{l, i, t}}}.
\end{align}
```
- 证据位置：Section 6 + Notion 公式块。

#### 设计 3：成熟阈值保护 + 输出权重置零
- 做了什么：
  - 新替换单元在 `m` 次更新前不参与再次替换（成熟阈值）。
  - 新单元输出权重置零，先避免破坏当前输出。
- 为什么这样设计：避免“刚替换即再替换”的抖动，并降低替换瞬间对已学功能的冲击。
- 解决的难点：难点 3。
- 关键公式 / 目标函数：算法流程以离散操作定义，核心超参数为 `ρ, m, η`。
- 证据位置：Algorithm 1 图示与正文解释。

![[assets/Maintaining Plasticity in Deep Continual Learning/image_013.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_014.png]]

### 4.3 训练 / 推理细节
- 训练阶段做了什么：
  - 反向传播更新；
  - 每步维护效用统计；
  - 选择并重初始化低效用单元。
- 推理阶段做了什么：与普通前馈网络一致，无额外推理模块。
- 损失函数组成：任务层面仍是常规分类/回归损失（如交叉熵）；CBP 不替代主损失，而是附加表征维护机制。
- 关键超参数：
  - `ρ`（replacement rate）
  - `m`（maturity threshold）
  - `η`（效用滑动平均衰减）
  - 任务学习率（step size）
- 复杂度 / 额外开销：
  - 新增 per-unit 统计与筛选开销；
  - 相对完整重训，代价较低。

---

## 5. 论文贡献（只写作者真正新增的东西）

- 贡献 1：系统且长时程地验证了现代深度学习在持续设置中的可塑性丧失，不只在单一网络/单一任务上展示现象。
- 贡献 2：提出了可解释的三类关联指标（死单元、权重幅度、有效秩）并给出一致证据链。
- 贡献 3：提出并验证 CBP，可在较宽参数范围维持可塑性，显著优于标准 backprop 及多种常见技巧。

---

## 6. 实验设置

- 数据集：
  - Continual ImageNet（二分类任务序列，任务对随机无放回抽样）
  - Online Permuted MNIST（800 任务）
  - Slowly-Changing Regression（控制分布缓慢漂移）
  - 正式版还含 CIFAR-100 与强化学习实验（来源：Nature）
- 模型 / 骨干网络：
  - Continual ImageNet：三层卷积+池化+三层全连接（末层 2 输出）。
  - Permuted MNIST：三隐藏层全连接网络（常见设置每层 2000 单元）。
  - Slowly-Changing：目标网络与学习网络分离设置（学习网络容量更小）。
- 对比方法：
  - Backpropagation（基线）
  - L2 regularization
  - Shrink and Perturb（S&P）
  - Dropout
  - Online Normalization
  - Adam
  - Continual Backpropagation（CBP）
- 评价指标：
  - 任务准确率 / 回归误差
  - Dead Units %
  - Weight Magnitude
  - Effective Rank
- 实现设置：
  - Continual ImageNet 任务训练后测试，报告多次独立运行平均与误差带。
  - Permuted MNIST 在线逐样本学习。
- 关键超参数：
  - 学习率（多档步长）
  - L2 权重衰减系数
  - S&P 噪声方差
  - Dropout 概率
  - CBP 替换率 `ρ` 和成熟阈值 `m`
- 是否开源代码 / 模型：是（GitHub）
- 实验是否公平（初步判断）：同一任务流、同网络骨架下横向对比较完整；不同方法对超参数敏感度不一致，解释时需保守。

![[assets/Maintaining Plasticity in Deep Continual Learning/image_002.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_004.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_008.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_019.png]]

---

## 7. 主要结果

### 7.1 主结果
- 结果 1（Continual ImageNet）：
  - 标准反向传播前期可达高性能，后期明显回落。
  - 预印本摘要给出代表值：早期任务约 89% 准确率，至第 2000 任务降至约 77%，接近线性基线。
- 结果 2（Permuted MNIST）：
  - 随任务推进，准确率普遍下降，且不同步长/宽度/切换速率下趋势一致。
- 结果 3（缓解方法对比）：
  - L2 与 S&P 可明显缓解；
  - Dropout、Adam、归一化在该持续场景下常显著恶化可塑性；
  - CBP 效果最稳健。

![[assets/Maintaining Plasticity in Deep Continual Learning/image_003.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_005.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_010.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_015.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_017.png]]

### 7.2 从结果中能读出的结论
- 结论 1：可塑性丧失在多任务、长时程持续训练下是系统性现象，不是偶然超参数失败。
- 结论 2：仅依赖梯度下降不足以长期维持可塑性；持续随机性注入是关键机制之一。
- 结论 3：一次性训练里常见的“好方法”并不自动迁移到持续学习。

### 7.3 最关键的证据
- 最关键表格：
  - 网络结构表（image_002）
  - 超参数搜索表（image_019）
- 最关键图：
  - Continual ImageNet 长时程下降曲线（image_003）
  - 各方法比较与三指标分解（image_012、image_016）
  - CBP 在长任务上保持优势（image_017）
- 最关键数字：
  - `89% -> 77% @ task 2000`（预印本摘要）
  - ReLU 设置下死单元在 800 任务后可达约 25%（文中描述）
- 为什么它最关键：
  - 同时覆盖“是否退化、为何退化、如何缓解”三层证据。

![[assets/Maintaining Plasticity in Deep Continual Learning/image_011.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_012.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_016.png]]

---

## 8. 消融实验

- 消融点 1：不同优化/正则方法比较（Backprop, L2, S&P, Dropout, Online Norm, Adam）
  - 改了什么：替换训练方法，保持任务和骨架一致。
  - 结果如何：S&P、L2 明显优于基线；Adam 与 Dropout 退化显著。
  - 说明了什么：可塑性维护机制比“常规优化器优势”更关键。

- 消融点 2：CBP 替换率 `ρ` 扫描
  - 改了什么：在 `1e-6 / 1e-5 / 1e-4` 等范围调整替换速度。
  - 结果如何：在较宽范围内都能维持高表现，过大/过小会影响最优点。
  - 说明了什么：CBP 对关键超参数有可用稳定区间。

- 消融点 3：慢变化回归中的激活函数与步长
  - 改了什么：tanh/sigmoid/ELU/ReLU/Leaky-ReLU/Swish + 多步长。
  - 结果如何：多数组合后期误差上升；ELU较缓但不根治。
  - 说明了什么：仅换激活函数不能根治可塑性问题。

![[assets/Maintaining Plasticity in Deep Continual Learning/image_006.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_007.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_009.png]]
![[assets/Maintaining Plasticity in Deep Continual Learning/image_018.png]]

---

## 9. 和已有工作的关系

- 这篇论文最接近哪些方法：
  - 持续学习中的稳定性方法（抗遗忘）
  - Shrink-and-Perturb、生成-测试式表示搜索
  - 现代优化器与正则化技术比较研究
- 和已有方法相比，最大的不同：
  - 把“可塑性”作为独立核心问题系统验证，而非仅把性能下降归因于遗忘。
- 真正的新意在哪里：
  - 给出从现象、指标到算法干预的闭环；
  - CBP 将“持续初始化”融入标准反向传播流程。
- 哪些地方更像“工程改进”而不是“方法创新”：
  - 某些超参数扫描与具体训练配方属于工程细化。
- 这篇论文在整个研究脉络里的位置：
  - 把持续学习从“记住过去”扩展到“保持未来可学”，是持续学习基础问题的关键补充。

---

## 10. 我的理解（这一节不能照抄论文）

### 10.1 直观理解
- 用自己的话解释这篇方法：
  - 网络一开始“可塑”是因为初始化带来多样、未饱和且小尺度权重；
  - 连续训练把这种状态磨平，单元变死、权重变大、表征维度塌缩；
  - CBP 相当于持续给网络做“局部新生”，把一小部分最不活跃结构重启。
- 它本质上像在做什么：
  - 把“优化”与“结构更新”交替进行，形成微量进化式维护机制。

### 10.2 我认为最关键的设计
- 最关键设计：低效用单元选择性替换（而不是随机替换全部或大量替换）。
- 为什么我觉得它最关键：它在“维持可塑性”和“保持当前功能”之间建立了可控平衡。

### 10.3 我认为最强的一点
- 把常见方法在持续设置下的优劣系统拉齐比较，结论直接可指导工程实践。

### 10.4 我认为最可疑的一点
- 目前效用定义仍是启发式，是否最优、是否跨架构最稳健仍需更多理论与大模型实证。

---

## 11. 局限性

- 局限 1：不同任务范式（超大规模预训练、LLM 对齐、多模态连续部署）上的外推仍待验证。
- 局限 2：CBP 引入额外状态与替换逻辑，工程系统中的并行实现和调参复杂度上升。
- 局限 3：论文核心实验多在受控基准，真实生产数据流的噪声与分布漂移更复杂。

---

## 12. 对我的启发

- 能直接借鉴的部分：
  - 在持续训练监控中新增三项健康指标：死单元比例、权重均值幅度、有效秩。
  - 用小比例结构重置替代“大规模周期重训”。
- 不能直接照搬的部分：
  - 效用公式与替换策略可能需按具体模型（如 Transformer）重写。
- 对我当前课题的启发：
  - “训练不再提升”可拆成“目标不对/优化受阻/表征塌缩”三类排查。
- 可以尝试的改进方向：
  - 将 CBP 与参数高效微调（LoRA/Adapter）结合。
  - 将替换触发改为自适应（基于性能斜率或秩变化）。
- 可以作为 baseline / 对比项 / ablation 的部分：
  - Backprop vs L2 vs S&P vs Adam vs CBP。

---

## 13. 待验证问题

- [ ] 问题 1：CBP 在 Transformer/LLM 持续微调中是否仍保持同样稳定区间？
- [ ] 问题 2：效用度量是否可由学习到的元策略替代手工公式？
- [ ] 问题 3：在带回放缓冲的强化学习中，CBP 与 replay 比例如何协同最优？

---

## 14. 一句话总结

- 这项工作证明了标准深度学习在持续场景会“越学越学不动”，而通过持续、选择性地重启低效单元可以长期维持可塑性。

---

## 15. 快速索引（便于二次回看）

- 核心公式：
  - 有效秩 `erank` 定义；
  - CBP 的贡献效用、均值修正贡献、总体效用递推式。
- 核心图表：
  - Continual ImageNet 退化图：`image_003`
  - 指标分解（dead units / weight / rank）：`image_011`, `image_016`
  - CBP 算法图：`image_014`
  - 5000 任务结果图：`image_017`
- 最值得复看的章节：
  - `4`（方法机制）
  - `7`（主证据）
  - `8`（超参数与消融）
- 复现时最需要注意的点：
  - 任务构造一致性
  - 替换率与成熟阈值联动
  - 三类健康指标的同步监控

### 15.1 整合说明 / 索引

- 本页已将 Notion 原始内容拆解归位到 `1~14`，包括背景叙述、实验细节、公式、超参数说明、讨论与附录信息。
- 19 张原始图片均已本地化并按语义分散插入到对应章节（方法图在方法节、结果图在结果节、附录图在消融/设置节）。
- 未保留“原始笔记整段转录”式堆放区。

### 15.2 导入来源与完整性记录

- Notion 原页面：
  - https://saputello.notion.site/Maintaining-Plasticity-in-Deep-Continual-Learning-1a184951e9628038a147e99f49b82bd1
- 源页面 ID：`1a184951-e962-8038-a147-e99f49b82bd1`
- 抓取块数量：`294`
- 图片本地化：`19/19`
- 原始 JSON：`notes/_notion_raw/Maintaining Plasticity in Deep Continual Learning.json`
- arXiv 预印本（联网校验）：https://arxiv.org/abs/2306.13812
- Nature 正式版（联网校验）：https://www.nature.com/articles/s41586-024-07711-7
- 代码仓库（联网校验）：https://github.com/shibhansh/loss-of-plasticity
- arXiv LaTeX 源码包（已下载）：
  - `papers_sources/Maintaining Plasticity in Deep Continual Learning/arxiv_2306.13812_source.tar.gz`
  - 已解压目录同路径下。
- 元信息校正说明：
  - Notion 页面使用预印本标题（Maintaining...）。
  - 正式期刊版标题为 Loss of plasticity...；作者列表较预印本更完整（Nature 页面显示新增作者）。

### 15.3 完成前自检（逐条）

- 1. 原始笔记所有内容都完整详细地进模板正文了吗？：是
- 2. 图片都插入到相应位置了吗？：是
- 3. 是否已联网补充并校正论文关键信息？：是
- 4. 是否还有内容残留在“原始堆放区/模板补充细节”而未拆入正文？：是
- 5. 若以上全是，才允许删除旧堆放内容并进入下一篇：是
- 6. 未出现“详见本页正文迁移小节 / 见 3.1 与 15.1”类索引语：是


## Wiki 关联

- 参考摘要：[[references/Maintaining Plasticity in Deep Continual Learning|Maintaining Plasticity in Deep Continual Learning]]
- 概念锚点：[[concepts/Loss of Plasticity in Continual Learning]]、[[concepts/Continual Backpropagation]]、[[concepts/Selective Neuron Reinitialization]]
- 实体锚点：[[entities/Shibhansh Dohare]]、[[entities/Qingfeng Lan]]、[[entities/Richard S. Sutton]]
- 综合页面：[[synthesis/Efficient Adaptation and Plasticity Retention]]、[[synthesis/Representation Capacity and Effective Rank]]、[[synthesis/Continual Learning Plasticity Maintenance Playbook]]
