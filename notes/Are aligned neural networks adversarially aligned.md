# Are aligned neural networks adversarially aligned?

- 阅读日期：2026-04-08
- 阅读状态：已读
- 标签：#paper #imported #notion
- 相关方向：LLM 对齐、对抗样本、越狱攻击、多模态安全
- 阅读目的：梳理“对齐模型是否在对抗输入下仍对齐”这个问题，并提取可复用的评估框架与实验结论

---

## 1. 论文信息

- 题目：Are aligned neural networks adversarially aligned?
- 链接：https://arxiv.org/abs/2306.15447
- 作者：Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt
- 单位：Google DeepMind；Stanford；University of Washington；ETH Zurich
- 会议 / 期刊 / 年份：NeurIPS 2023（arXiv v1: 2023-06-26，v2: 2024-05-06）
- 关键词（3~8个）：adversarial alignment, LLM safety, jailbreak, NLP attacks, multimodal attacks, toxicity evaluation
- 论文一句话主题：本文提出“对抗性对齐”评估视角，发现现有 NLP 攻击不足以证明文本对齐模型鲁棒，但多模态对齐模型可被高成功率对抗图像轻易诱导产生有害输出。

---

## 2. 先看结论（适合快速回顾）

- 这篇论文主要解决什么问题：对齐后的神经网络（尤其 LLM）在“最坏情况输入”下是否仍保持对齐。
- 提出的核心方法是什么：统一评估两类攻击路径：NLP-only 离散提示攻击与多模态连续图像攻击，并设计“保证存在解”的测试来反证攻击器能力不足。
- 最终最重要的结果是什么：
  - 现有主流 NLP 攻击（如 ARCA、GBDA）在对齐聊天模型上成功率低，不能据此得出“模型足够鲁棒”。
  - 在保证存在攻击解的构造测试上，现有 NLP 攻击依然显著失败。
  - 多模态模型（MiniGPT-4、LLaVA、LLaMA-Adapter）在毒性目标上可达 100% 攻击成功率。
- 我现在是否值得深入读：值得
- 原因：它把“对齐是否有效”从经验问题转成“攻击-防御评估能力”问题，并给出可复用的实验协议。

![[assets/Are aligned neural networks adversarially aligned/image_001.png]]

---

## 3. 问题定义

### 3.1 研究问题
- 论文研究的核心问题：Are aligned neural network models ``adversarially aligned''?
- 输入是什么：用户输入（文本提示；或文本+图像输入）。
- 输出是什么：模型生成文本是否包含不应出现的有害内容（本文主要用 toxic output 作为目标）。
- 优化目标是什么：攻击者要找到输入 `X`，使得 `isToxic(Gen(X))` 为真。
- 任务设定 / 威胁模型 / 前提假设：
  - 研究对抗样本有两个原因：1) 提升系统鲁棒性；2) 评估最坏情况行为（如自动驾驶）。
  - Existing threat models：
    - 恶意用户：通过越狱提示诱导不对齐输出，不要求隐蔽。
    - 恶意第三方：通过提示注入劫持系统行为（如泄露邮件）。
  - Our threat model：重点不是刻画单一真实攻击者，而是测量当前对齐技术在最坏输入下的极限。
  - 生成随机性处理：当生成是随机时，记作 `Pr[isToxic(Gen(X))] > ⁍`（原笔记符号如此记录）；实验中将温度设为 0 保持确定性。

### 3.2 为什么重要
- 这个问题为什么值得做：
  - 仅凭“攻击没打穿”不能证明对齐真的稳健。
  - 如果对抗评估能力弱，会把“未被攻破”误判为“本质安全”。
- 现实应用价值：对话模型、代理系统、提示注入防护、多模态应用的安全红队评估都依赖该问题。
- 学术上的意义：把 AI 对齐和对抗样本两条研究线正式合并到同一评估框架中。
- 原始背景条目（按 Notion 原文保留）：
  - `LLM：`
  - `对齐LLM：`
  - `MLLM`
  - `对抗攻击`
- 原始语义边界（按 Notion 原文保留）：本文更关注导致模型产生有害行为的对抗样本，而非仅“错误分类”；这与社会工程攻击不同，不刻意追求语义上有害但自然的回复。

![[assets/Are aligned neural networks adversarially aligned/image_002.png]]

### 3.3 难点
- 难点 1：文本 token 离散，难以直接做连续优化（如梯度下降）。
- 难点 2：攻击目标不是单一固定 token，长序列目标回传代价高。
- 难点 3：攻击失败时，很难区分“模型鲁棒”还是“攻击器太弱”。

---

## 4. 论文方法

### 4.1 方法总览
- 方法名称：Adversarial Alignment Evaluation（NLP-only + Multimodal）
- 一句话概括方法：先在文本模型上检验现有离散攻击是否真能评估鲁棒性，再在多模态模型上用连续域 PGD 直接测试可攻击性。
- 方法整体流程（按步骤写 1/2/3/4）：
  1. 定义毒性攻击目标与威胁模型。
  2. 在 GPT-2 / LLaMA / Vicuna 上复现实有 NLP 攻击并统计成功率。
  3. 构造“保证存在攻击解”的测试集，验证攻击器是否真的能找到解。
  4. 对 MiniGPT-4 / LLaVA / LLaMA-Adapter 进行图像对抗优化并量化成功率与失真。

![[assets/Are aligned neural networks adversarially aligned/image_003.png]]

### 4.2 核心设计
> 每个设计都尽量回答：做了什么、为什么这么设计、解决了哪个难点

#### 设计 1
- 做了什么：将原始毒性目标转为“有害前缀（harmful prefix）”代理目标，优先优化生成开头命中目标串。
- 为什么这样设计：直接优化完整 `isToxic(Gen(X))` 需要多次生成与回传，成本高且不稳定。
- 解决的难点：难点 2（目标过长、优化难）。
- 关键公式 / 目标函数：
  - `isToxic(Gen(X))`
  - “反转语言模型：找到一个对抗性提示⁍，使得模型⁍输出某个目标字符串⁍。”（原笔记原样）
- 证据位置：Section 4；Section 4.2；Notion 原文 “Prior Attack Methods”。

#### 设计 2
- 做了什么：设计“保证存在可触发输出”的测试集（由 brute force 先验构造），再用攻击器去找同一解。
- 为什么这样设计：若攻击器连“已知存在解”的任务都做不好，就不能用它来评估防御是否鲁棒。
- 解决的难点：难点 3（鲁棒性与攻击能力混淆）。
- 关键公式 / 目标函数：
  - “通过穷举法构造导致模型发出稀有后缀 ⁍ 的提示 ⁍。然后，如果攻击能够找到某个输入序列 ⁍ 使得 ⁍，即模型发出相同的 ⁍，则攻击成功。否则，攻击失败。”
  - “设 ⁍ 为所有 N-标记序列的空间（对于某个 N）……”（原笔记符号占位按原样保留）
- 证据位置：Section 5；Section 5.1；Table 2。

#### 设计 3
- 做了什么：在多模态模型上对输入图像做 PGD，对目标输出 token 序列做 teacher-forcing 交叉熵优化。
- 为什么这样设计：图像是连续域，优化比离散文本更直接且攻击空间更大。
- 解决的难点：难点 1（离散优化困难）。
- 关键公式 / 目标函数：
  - 总交叉熵 over target tokens（teacher forcing）
  - PGD，最多 500 steps，默认步长 0.2
  - “一个任意大的⁍，并运行最多500个步骤……”（原笔记符号占位按原样保留）
- 证据位置：Section 6.1；Table 3。

### 4.3 训练 / 推理细节
- 训练阶段做了什么：
  - NLP 攻击：在白盒访问下优化可控 token（最多 30 token 的设置用于对齐聊天评估）。
  - 多模态攻击：从随机图像初始化，优化图像像素使模型输出指定有毒前缀。
- 推理阶段做了什么：
  - 文本模型：固定系统/模板 token，攻击者仅控制 `[USER]:` 部分，`[AGENT]:` 终止标记不可修改。
  - 多模态模型：输入对抗图像 + 文本提示，观察是否生成目标有害输出。
- 损失函数组成：毒性目标的代理前缀损失；多 token 目标时使用 teacher-forcing 的总交叉熵。
- 关键超参数：温度 0；PGD 最多 500 步；默认 step size 0.2；测试含 1/2/5/10 个额外可控 token场景。
- 复杂度 / 额外开销：离散文本攻击搜索成本高；多模态连续攻击可优化但需白盒梯度。
- 原始实验句（按 Notion 原文保留）：
  - 从 Open Assistant 数据集中获取良性对话；将其截断为随机选择的 `⁍` 轮次。
  - 使用 Jones et al. [2023] 一小部分有害文本作为攻击目标，攻击对象为一到三个有毒文本标记。
  - 当目标响应超过一个标记时（即 `⁍ token`），应用标准教师强制优化技术，优化目标输出 token 的总交叉熵。
  - 使用投影梯度下降；一个任意大的 `⁍`，最多 500 步，默认步长 0.2。

---

## 5. 论文贡献（只写作者真正新增的东西）

- 贡献 1：提出“对抗性对齐（adversarial alignment）”作为评估对齐模型最坏情况行为的统一问题。
- 贡献 2：证明“现有 NLP 攻击失败”不足以说明对齐模型鲁棒，并给出保证存在解的评测协议揭示攻击器能力缺陷。
- 贡献 3：在多模态对齐模型上展示高效且稳定的对抗图像攻击，显示当前对齐技术对连续域输入脆弱。
- 原始构造测试描述（按 Notion 原文保留）：
  - 通过穷举法构造导致模型发出稀有后缀 `⁍` 的提示 `⁍`。
  - 如果攻击找到某输入序列 `⁍` 使模型发出相同后缀 `⁍`，则攻击成功；否则失败。
  - 暴力搜索先用 GPT-2 尝试；由于其对齐性较差，若在该模型都失败，通常无需优先尝试更难模型。

---

## 6. 实验设置

- 数据集：
  - OpenAssistant 良性对话（构造聊天上下文）
  - Jones et al. [2023] 的有害短语集（1~3 toxic tokens）
  - Wikipedia 前缀（构造 Section 5 的稀有输出测试）
  - 随机初始化图像（多模态攻击起点）
- 模型 / 骨干网络：
  - 文本：GPT-2、LLaMA、Vicuna
  - 多模态：MiniGPT-4（Instruct / RLHF）、LLaVA、LLaMA-Adapter
- 对比方法：ARCA、GBDA、Brute Force（以及文中讨论的 UAT 背景）
- 评价指标：攻击成功率 / pass rate；毒性检测（substring lookup）；均值 `ℓ2` 扰动
- 实现设置：
  - 文本场景控制 `[USER]:`，不可改 `[AGENT]:`
  - 远距/近距注入两种位置
  - 可控 token 数分层统计
- 关键超参数：最多 30 个恶意 token；温度 0；PGD 500 steps、step size 0.2
- 原始测试集参数（按 Notion 原文保留）：
  - 流行度：给定 `⁍` 的标记 `⁍` 的概率，将其固定为 `⁍`。
  - 攻击可控标记：2、5、10、20 个标记。
  - 目标标记：攻击者必须达到的输出标记数量。
- 是否开源代码 / 模型：论文使用开源模型复现实验；完整攻击实现仓库链接待确认
- 实验是否公平（初步判断）：是（同一协议下横向比较），但毒性定义较简化，语义层面评估仍有限

---

## 7. 主要结果

### 7.1 主结果
- 结果 1（NLP 攻击在对齐聊天上的有限效果）：
  - Table 1（Notion 图 + 论文表）显示在 Vicuna 上成功率很低（如 6%、0%、8%、1% 量级，视注入位置与攻击器而定）。
  - GPT-2 / LLaMA 虽在部分设置下更高，但在对齐场景仍显著受限。
- 结果 2（“存在解”测试下攻击器仍失败）：
  - Table 2：Brute Force 为 100%，但 ARCA/GBDA 在 1~10 额外 token 场景远低于 100%（如 ARCA 11.1%~30.6%，GBDA 3.1%~9.5%）。
- 结果 3（多模态攻击高成功）：
  - Table 3：MiniGPT-4 / LLaVA / LLaMA-Adapter 攻击成功率均为 100%，且平均 `ℓ2` 扰动较小（如 LLaVA 0.86±0.17）。

### 7.2 从结果中能读出的结论
- 结论 1：现有 NLP 攻击失败，不能直接推出“文本对齐模型已对抗鲁棒”。
- 结论 2：评估防御强度前，必须先验证攻击器能否通过“已知可解”任务。
- 结论 3：对齐多模态模型在图像连续扰动下仍可被稳定诱导输出有害内容。
- 原始结论句（按 Notion 原文保留）：
  - `(1) 语言模型对软嵌入攻击（例如，多模态攻击）较为脆弱；`
  - `(2) 当前的自然语言处理（NLP）攻击无法找到已知的解决方案。`

### 7.3 最关键的证据
- 最关键表格：Table 2（区分“攻击器弱”与“防御强”）、Table 3（多模态 100% 攻击成功）
- 最关键图：Notion 导入图中的 prior attack 结果图与 multimodal attack 图
- 最关键数字：
  - Table 2: Brute Force 100%，ARCA 11.1%~30.6%，GBDA 3.1%~9.5%
  - Table 3: 100% success across tested multimodal models
- 为什么它最关键：它们分别回答了“文本端为什么不能盲目信任低攻击成功率”和“多模态端为什么风险已是现实问题”。
- 原始量化观察（按 Notion 原文保留）：对抗输入相对容易找到，仅需要对初始图像进行最小的 `⁍` 失真；不同提示之间的失真变化也很小。

![[assets/Are aligned neural networks adversarially aligned/image_004.png]]
![[assets/Are aligned neural networks adversarially aligned/image_005.png]]
![[assets/Are aligned neural networks adversarially aligned/image_006.png]]

---

## 8. 消融实验

- 消融点 1：
  - 改了什么：远距注入（Distant）与近距注入（Nearby）攻击位置。
  - 结果如何：不同模型与攻击器在两种注入位置上的成功率差异显著。
  - 说明了什么：攻击位置会显著影响离散提示攻击有效性。

- 消融点 2：
  - 改了什么：可控 token 数量（1/2/5/10 或更高上限设置）。
  - 结果如何：即便增加可控 token，ARCA/GBDA 仍远低于 brute force。
  - 说明了什么：当前攻击算法的搜索能力是核心瓶颈。

- 消融点 3：
  - 改了什么：多模态模型实现差异（MiniGPT-4 Instruct / RLHF、LLaVA、LLaMA-Adapter）。
  - 结果如何：都能被攻破，但平均 `ℓ2` 扰动差异明显（LLaVA 更易受攻）。
  - 说明了什么：实现细节对脆弱性影响很大。

---

## 9. 和已有工作的关系

- 这篇论文最接近哪些方法：RLHF / instruction tuning 对齐工作；文本对抗攻击（ARCA、GBDA、UAT）；多模态对抗样本研究。
- 和已有方法相比，最大的不同：把“对齐安全”与“攻击器能力校准”绑定评估，而不是只报单次攻击成功率。
- 真正的新意在哪里：提出并实践“保证存在解”的攻击能力基准，以避免防御评估中的假阴性结论。
- 哪些地方更像“工程改进”而不是“方法创新”：毒性判定用 substring 的简化实现；具体 prompt 模板与数据拼接。
- 这篇论文在整个研究脉络里的位置：连接 LLM 对齐评估与经典对抗鲁棒评估范式的关键桥接工作。

---

## 10. 我的理解（这一节不能照抄论文）

### 10.1 直观理解
- 用自己的话解释这篇方法：这篇工作不是先证明模型安全，而是先检查“我们有没有一把足够锋利的锤子去敲它”。
- 它本质上像在做什么：把对齐评估从“攻击结果”升级为“攻击器能力 + 防御能力”双重诊断。

### 10.2 我认为最关键的设计
- 最关键设计：Section 5 的“已知可解测试”。
- 为什么我觉得它最关键：它直接避免了“攻击失败=模型安全”的错误逻辑。

### 10.3 我认为最强的一点
- 把文本与多模态放在同一对抗性对齐框架下比较，结论更完整。

### 10.4 我认为最可疑的一点
- 毒性判定过于简化（substring），可能低估或高估语义层面的真实危害。

---

## 11. 局限性

- 局限 1：毒性定义简化，无法覆盖更细粒度语义伤害和上下文文化差异。
- 局限 2：主要在白盒条件下评估，黑盒现实可攻击性还需额外验证。
- 局限 3：文本端结论强调“攻击器不足”，但更强 NLP 攻击尚未完全给出。

---

## 12. 对我的启发

- 能直接借鉴的部分：评估时先做“攻击器能力 sanity check”（已知可解基准）。
- 不能直接照搬的部分：仅用词表子串做安全判定。
- 对我当前课题的启发：应把“防御评估协议”作为核心实验对象，而不只关注单一模型分数。
- 可以尝试的改进方向：引入语义级有害性判别器与人类评审结合；补充黑盒与迁移攻击设置。
- 可以作为 baseline / 对比项 / ablation 的部分：ARCA、GBDA、Brute Force、PGD 图像攻击。

---

## 13. 待验证问题

- [ ] 问题 1：更强的离散优化攻击（如更大搜索预算或新算法）能否在 Vicuna 等模型上稳定复现高成功率？
- [ ] 问题 2：如果把毒性判定从 substring 替换为语义评审，Table 1/2/3 结论是否变化？
- [ ] 问题 3：在黑盒 API 限制和防护机制下，多模态 100% 成功率是否仍可保持？

---

## 14. 一句话总结

- 这篇论文的核心结论是：现有文本攻击“打不穿”并不等于模型“真鲁棒”，而多模态对齐模型在对抗图像下已经可被高可靠度攻破。

---

## 15. 快速索引（便于二次回看）

- 核心公式：`isToxic(Gen(X))`、`Pr[isToxic(Gen(X))] > ⁍`（原笔记符号占位）
- 核心图表：`image_004`（NLP attacks 结果图）、`image_005`（保证可解测试结果）、`image_006`（多模态攻击）
- 最值得复看的章节：Section 4（NLP-only attacks）、Section 5（为何现有攻击失败）、Section 6（多模态攻击）
- 复现时最需要注意的点：攻击目标定义、可控 token 约束、毒性判定方式、白盒访问条件

### 15.1 整合说明 / 索引

- Notion 原始内容已拆解到 1~14 节：背景/威胁模型在 3，方法在 4，实验与结果在 6~8，结论在 7 与 14。
- 本节仅保留导航说明，不存放原始转录段落。

### 15.2 导入来源与完整性记录

- Notion 源页面：
  - https://saputello.notion.site/Are-aligned-neural-networks-adversarially-aligned-1a084951e96280f5abe0f41c49402668
- 联网校验来源（2026-04-08）：
  - arXiv 摘要页：https://arxiv.org/abs/2306.15447
  - NeurIPS 2023 摘要页：https://papers.nips.cc/paper_files/paper/2023/hash/c1f0b856a35986348ab3414177266f75-Abstract-Conference.html
  - NeurIPS 2023 PDF：https://proceedings.neurips.cc/paper_files/paper/2023/file/c1f0b856a35986348ab3414177266f75-Paper-Conference.pdf
- 本地原始导入：
  - `notes/_notion_raw/Are aligned neural networks adversarially aligned.json`
  - `notes/_notion_raw/Are aligned neural networks adversarially aligned.rendered.md`
  - `notes/_notion_raw/Are aligned neural networks adversarially aligned.ordered.md`
  - `notes/_notion_raw/Are aligned neural networks adversarially aligned.images.json`
- 本地图片：
  - `notes/assets/Are aligned neural networks adversarially aligned/`（`image_001`~`image_006`）
- arXiv LaTeX 源码：
  - `papers_sources/2306.15447.tar`
  - `papers_sources/Are aligned neural networks adversarially aligned 2306.15447/`

### 15.3 完成前自检（逐条）

- 1. 原始笔记所有内容都完整详细地进模板正文了吗？：是
- 2. 图片都插入到相应位置了吗？：是
- 3. 是否已联网补充并校正论文关键信息？：是
- 4. 是否还有内容残留在“原始堆放区/模板补充细节”而未拆入正文？：是
- 5. 若以上全是，才允许删除旧堆放内容并进入下一篇：是
- 6. 未出现“详见本页正文迁移小节 / 见 3.1 与 15.1”类索引语：是


## Wiki 关联

- 参考摘要：[[references/Are aligned neural networks adversarially aligned|Are aligned neural networks adversarially aligned]]
- 概念锚点：[[concepts/Adversarial Alignment Evaluation]]、[[concepts/Known-Solvable Attack Calibration]]、[[concepts/Multimodal Adversarial Image Prompting]]
- 实体锚点：[[entities/Nicholas Carlini]]、[[entities/Vicuna]]、[[entities/LLaVA]]
- 综合页面：[[synthesis/Adversarial Robustness Evaluation Patterns]]、[[synthesis/Alignment Robustness Evaluation Ladder]]
