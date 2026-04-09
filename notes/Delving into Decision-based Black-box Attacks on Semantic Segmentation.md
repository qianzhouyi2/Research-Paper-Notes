---
title: Delving into Decision-based Black-box Attacks on Semantic Segmentation
category: note
tags:
  - paper
  - semantic-segmentation
  - black-box-attack
  - adversarial-attack
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh.md
created: 2026-04-06
updated: 2026-04-09
summary: DLA 论文阅读笔记，覆盖问题定义、方法细节、实验结果与可复用实现思路。
provenance:
  extracted: 0.82
  inferred: 0.16
  ambiguous: 0.02
---

# Delving into Decision-based Black-box Attacks on Semantic Segmentation

- 阅读日期：[[2026-04-06]]
- 阅读状态：已读
- 标签：#paper #semantic-segmentation #black-box-attack #adversarial-attack
- 相关方向：决策型黑盒攻击、语义分割鲁棒性
- 阅读目的：整理 DLA 的可复现攻击流程与实验证据

---

## 1. 论文信息

- 题目：Delving into Decision-based Black-box Attacks on Semantic Segmentation
- 链接：https://arxiv.org/abs/2402.01220
- 作者：Zhaoyu Chen, Zhengyang Shan, Jingwen Chang, Kaixun Jiang, Dingkang Yang, Yiting Cheng, Wenqiang Zhang
- 单位：复旦大学相关实验室
- 发表：arXiv 2402.01220（2024）
- 关键词：decision-based black-box attack, semantic segmentation, adversarial robustness

### 1.1 关联页面

- Reference：[[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- Concepts：[[concepts/Deep Neural Network (DNN)]]、[[concepts/Semantic Segmentation]]、[[concepts/Decision-based Black-box Attack for Segmentation]]、[[concepts/Indirect Local Attack in Segmentation]]、[[concepts/SegPGD]]、[[concepts/Proxy Index mIoU Optimization]]、[[concepts/Discrete Linear Noise]]、[[concepts/Natural Evolutionary Strategies (NES)]]、[[concepts/L-infinity Norm Ball]]、[[concepts/Perturbation Interaction]]、[[concepts/Query-Efficient Attack Evaluation]]
- Synthesis：[[synthesis/Decision-based Segmentation Attack Landscape]]
- Entities：[[entities/Zhaoyu Chen]]、[[entities/Wenqiang Zhang]]、[[entities/FCN]]、[[entities/PSPNet]]、[[entities/DeepLabv3]]、[[entities/SegFormer]]、[[entities/MaskFormer]]

---

## 2. 先看结论（适合快速回顾）

- 这篇论文主要解决什么问题：仅有分割标签输出时，如何高效攻击语义分割模型。
- 提出的核心方法是什么：DLA（Discrete Linear Attack），以 mIoU 下降为单调优化目标。
- 最终最重要的结果是什么：在有限查询预算下实现更优攻击效果与效率。
- 我现在是否值得深入读：值得
- 原因：任务定义清晰，攻击流程可直接实现。

---

## 3. 问题定义

### 3.1 研究问题

- 论文关注的是：在只能拿到语义分割标签输出、拿不到概率和梯度的前提下，如何高效攻击语义分割模型。
- 作者指出，这个问题和图像分类下的 decision-based attack 不一样，原因在于语义分割是像素级、多约束优化，攻击难度明显更高。
- 目标不是单纯让整张图“误分类”，而是在查询预算有限的情况下尽可能降低分割结果的 mIoU。

### 3.2 为什么重要

- 这个问题为什么值得做：语义分割模型在真实场景部署广泛，黑盒鲁棒性评估需求强。
- 现实应用价值：可用于安全评测、防御基线和鲁棒训练闭环。
- 学术上的意义：扩展 decision-based attack 到像素级结构化输出任务。

### 3.3 难点

- 不一致的优化目标：分类任务里常见做法是先越过决策边界，再尽量缩小扰动；但在分割里，扰动变小往往会让 mIoU 回升，所以“减小扰动”和“维持攻击效果”并不一致。
- 扰动相互干扰：一次更新新扰动后，前一次已经被攻击成功的像素可能恢复正确类别，导致优化过程反复来回。
- 参数空间太复杂：分割任务对每个像素都有约束，本质上比图像分类更像高维多约束搜索，普通随机扰动或零阶优化都很难高效。

---

## 4. 论文方法

### 4.1 方法总览

- 作者提出了 DLA（Discrete Linear Attack），这是论文的核心方法。
- 整体思路不是去逼近分类边界，而是从干净图像出发，直接优化一个代理指标（proxy index），让每次查询都朝着“更差的分割结果”推进。
- 文中比较了 PAcc 和 mIoU 两个代理指标，最后选择 mIoU，因为它更能反映整体分割质量，也更适合作为优化目标。

### 4.2 核心设计
> 每个设计都尽量回答：做了什么、为什么这么设计、解决了哪个难点

### 0. 先看清 DLA 实际在优化什么

- 论文里的真实目标是让尽可能多的像素被错分，即
  `argmax_δ Σ 1(f(x+δ)_i != y_i)`，并满足 `||δ||_∞ <= ε`。
- 但在 decision-based 设置下，模型只返回每个像素的类别标签，不返回概率，所以 DLA 不直接做梯度或边界逼近，而是用一个可比较的代理指标 `L(x')` 来决定“这次更新要不要收下”。
- 论文最终选的是单张图像上的 `mIoU` 作为 `L(x')`。每次 query 后，只要新样本的 `mIoU` 更低，就接受这次更新；否则回退。
- 这点很关键：DLA 本质上是一个“基于 mIoU 单调下降准则的随机搜索”，不是传统 boundary attack 那种“先越界、再缩扰动”。
- 用来说明这一点的最小基线就是 Random Attack：从干净图像出发随机试探小扰动，只在代理指标下降时接受更新，属于 [[concepts/Proxy Index mIoU Optimization|proxy-guided accept/reject search]]。

### 1. 扰动是怎么表示的

- DLA 不是在整张图的每个像素上独立采样连续值，而是把扰动直接限制到 `{-ε, +ε}`，即把连续空间 `[-ε, +ε]^d` 压成离散空间。
- 这可以直观理解为：先把搜索限制在 [[concepts/L-infinity Norm Ball|`L_\infty` 范数球]] 的顶点集合里，因为这些顶点正对应“每一维都取到 `±ε`”。
- 更进一步，它不是采样 patch，也不是采样 full-image random noise，而是采样 **discrete linear noise**。
- 方法定义里，线性噪声只有一个空间维度的自由度：
  `δ ~ {-ε, +ε}^h` 或 `δ ~ {-ε, +ε}^w`。
- 这意味着可以把它理解成：
  `horizontal` 时，先为每一行采一个 `±ε`，再沿整行展开，形成横向条纹；
  `vertical` 时，先为每一列采一个 `±ε`，再沿整列展开，形成纵向条纹。
- 从 `main.tex` 的公式和算法写法看，作者主要压缩的是空间维度，而不是让每个像素、每个通道都单独自由变化。这里我做的是基于源码写法的实现层推断。
- 这样做的直接好处有两个：
  1. 参数维度从“每个像素一个值”降成“每行或每列一个符号”。
  2. 扰动形状比 patch 更细长，不容易出现特别显眼的彩色块，同时不同更新之间也更不容易互相覆盖。

### 2. 方法流程（非伪代码版）

- DLA 把攻击分成“探索 + 校准”两段：先找到有效扰动方向，再进行局部细化。
- 探索阶段交替试 horizontal / vertical 的离散线性噪声，用 `mIoU` 下降作为接受准则，选出更有效的初始化扰动。
- 校准阶段在该初始化上做逐步细化的局部符号翻转，只保留能继续降低 `mIoU` 的更新。

### 3. 为什么这个设计比普通随机搜索更有效

- 它先用 `mIoU` 把“有没有进步”定义清楚了，避免了 segmentation 下目标函数和边界缩扰动目标不一致的问题。
- 它把连续高维空间压成了离散线结构，所以前期 exploration 的 query 利用率更高。
- 它后面不是重新采样，而是围绕当前最好扰动做局部符号翻转，因此更像“离散坐标搜索”，不会每一步都把前面好不容易找到的结构打散。
- calibration 沿着单一方向做层次化切分，也是在进一步控制搜索复杂度，这和直接做二维 patch 级别的任意更新不同。

### 4.3 训练 / 推理细节

- 训练阶段做了什么：无训练（攻击方法）。
- 推理阶段做了什么：探索初始化噪声后进行 coarse-to-fine 校准搜索。
- 损失函数组成：以单图 `mIoU` 作为代理优化指标。
- 关键超参数：查询预算、`epsilon`、探索/校准比例、分块层数。
- 复杂度 / 额外开销：查询复杂度为核心约束。


### 4.4 代码实现对照（完整迁移）

#### 实现代码伪代码

- 下面这份伪代码对应当前 `SegmentationDLAAttack` 的 PyTorch 实现，是“代码实现版 DLA”，不完全等同于论文里的原始 Algorithm 1。
- 最大区别是：这份实现按 **单张图像** 逐个攻击，每一轮先各试一次 horizontal / vertical seed，再只沿当前更优方向做分层 flip refine，而不是严格按论文里的前 `T/5` exploration、后 `4T/5` calibration 来切预算。

### 1. 单张图像 mIoU 计算

```text
function PER_IMAGE_MIOU(pred, target, ignore_index):
    flatten pred and target to [B, HW]
    valid = (target != ignore_index) if ignore_index exists else all True
    if all positions are invalid:
        return zeros(B)

    build per-image confusion matrix with bincount
    for each image:
        intersection = diagonal(confusion)
        union = row_sum + col_sum - intersection
        only average classes with union > 0
        miou = mean(intersection / union over present classes)
    return per-image miou
```

### 2. 线性噪声采样

```text
function RANDOM_LINEAR_NOISE(C, H, W, axis):
    if axis == horizontal:
        sample H row signs from {-1, +1}
        expand each row sign across the full width and all channels
    else if axis == vertical:
        sample W column signs from {-1, +1}
        expand each column sign across the full height and all channels
    return noise
```

### 3. 分层翻转细化

```text
function REFINE_NOISE(model, x, y, base_noise, axis, current_best_miou, queries_left):
    current_noise = clone(base_noise)
    best_noise = clone(base_noise)
    current_miou = current_best_miou
    best_miou = current_best_miou
    used_queries = 0

    dim = H if axis == horizontal else W
    max_level = ceil(log2(dim)) + 1

    for level in [0 .. max_level - 1]:
        num_segments = 2^level
        seg_len = ceil(dim / num_segments)

        for segment_idx in [0 .. num_segments - 1]:
            if used_queries >= queries_left:
                return best_noise, best_miou, used_queries

            [start, end) = current segment range
            if empty segment:
                continue

            flip sign of current_noise on this segment
            candidate = clamp(x + current_noise, 0, 1)
            candidate_miou = mIoU(model(candidate), y)
            used_queries += 1

            if candidate_miou <= current_miou:
                keep this flip
                current_miou = candidate_miou
                if candidate_miou < best_miou:
                    best_noise = clone(current_noise)
                    best_miou = candidate_miou
            else:
                revert this flip

    return best_noise, best_miou, used_queries
```

### 4. 主攻击流程

```text
function PERTURB(model, inputs, targets):
    set model to eval
    adv_inputs = clone(inputs)
    clean_pred = argmax(model(inputs))
    best_mious = PER_IMAGE_MIOU(clean_pred, targets)
    queries = zeros(batch_size)

    for each image i in batch:
        x = inputs[i]
        y = targets[i]
        best_input = x
        best_miou = best_mious[i]
        used_queries = 0

        while used_queries < total_query_budget:
            remaining = total_query_budget - used_queries

            best_seed_axis = None
            best_seed_noise = None
            best_seed_miou = +inf

            for axis in [horizontal, vertical]:
                if remaining <= 0:
                    break

                noise = epsilon * RANDOM_LINEAR_NOISE(C, H, W, axis)
                candidate = clamp(x + noise, 0, 1)
                candidate_miou = mIoU(model(candidate), y)
                used_queries += 1
                remaining = total_query_budget - used_queries

                if candidate_miou < best_miou:
                    best_input = candidate
                    best_miou = candidate_miou

                if candidate_miou < best_seed_miou:
                    best_seed_axis = axis
                    best_seed_noise = noise
                    best_seed_miou = candidate_miou

            if no valid seed or no remaining queries:
                break

            refined_noise, refined_miou, refine_queries =
                REFINE_NOISE(
                    model, x, y,
                    best_seed_noise,
                    best_seed_axis,
                    best_seed_miou,
                    remaining
                )
            used_queries += refine_queries

            if refined_miou < best_miou:
                best_input = clamp(x + refined_noise, 0, 1)
                best_miou = refined_miou

        adv_inputs[i] = best_input
        best_mious[i] = best_miou
        queries[i] = used_queries

    return AttackResult(
        adv_inputs = adv_inputs,
        objective = -best_mious,
        margin = -best_mious,
        queries = queries
    )
```

### 5. 这份实现和论文原始 DLA 的对应关系

- 对应论文里的 `proxy index`：代码里直接用 `_per_image_miou(...)`。
- 对应论文里的 `discrete linear noise`：代码里是 `_random_linear_noise(...)`，横向按行采样，纵向按列采样。
- 对应论文里的 `perturbation calibration`：代码里是 `_refine_noise(...)`，按 `2^level` 个分段逐层翻转符号。
- 对应论文里的“保留更优扰动”：代码里所有更新都以 `candidate_miou` 是否更低作为接受条件。
- 和论文不同的是，这个实现没有显式维护论文伪代码中的 `M / d / i / n` 这套状态变量，而是直接用当前 `axis`、`level`、`segment_idx` 和 `current_noise` 来完成同一类 coarse-to-fine flip 搜索
---

## 5. 论文贡献（只写作者真正新增的东西）

- 首次系统研究了 semantic segmentation 场景下的 decision-based black-box attack。
- 明确总结了该任务的三个核心困难：优化目标不一致、扰动交互、参数空间复杂。
- 提出了 DLA，用离散线性噪声和分层符号翻转实现高效攻击。
- 在多个数据集和模型上证明，DLA 在攻击强度和查询效率上都明显优于现有方法。

> 判断标准：如果删掉这一点，论文是否还成立？如果“是”，那它可能不是核心贡献。

---

## 6. 实验设置

- 数据集：Cityscapes、ADE20K
- 模型：FCN、PSPNet、DeepLabV3、SegFormer、MaskFormer
- 对比方法：Random、NES、Bandits、ZO-SignSGD、SignHunter、SimBA、Square Attack
- 评价指标：攻击后的 mIoU，越低说明攻击越成功
- 查询预算：主要报告 50 / 200 queries，另做了 10 queries 的极限实验
- 扰动预算：`epsilon = 8`

## 7. 主要结果

### 7.1 主结果

### 7.2 重要实验表格（Markdown）

- 说明：除 `Clean` 外，带斜杠的数值统一表示 `50 / 200 queries` 下的 mIoU，越低越好。
- 说明：消融表中的空白表示该设计没有启用。

### 1. 主结果：Cityscapes

| Attack     |           FCN |        PSPNet |     DeepLabV3 |       SegFormer |    MaskFormer |
| ---------- | ------------: | ------------: | ------------: | --------------: | ------------: |
| Clean      |         77.89 |         77.83 |         77.70 |           80.43 |         73.91 |
| Random     |   35.76/34.94 |   48.81/47.18 |   54.57/52.77 |     58.59/56.07 |   39.09/39.06 |
| NES        |   34.47/33.82 |   48.34/47.40 |   54.32/52.99 |     58.00/55.34 |   51.94/52.56 |
| Bandits    |   18.17/15.65 |   20.81/17.55 |   29.85/26.73 |     39.43/36.14 |   26.94/26.88 |
| ZO-SignSGD |   34.97/34.01 |   46.69/45.80 |   51.83/50.54 |     55.67/54.81 |   49.65/49.59 |
| SignHunter |   23.88/21.67 |   33.52/26.04 |   44.24/35.93 |     41.38/34.18 |   47.05/27.06 |
| SimBA      |   33.74/29.58 |   46.27/40.22 |   54.67/50.17 |     54.17/52.67 |   33.52/32.71 |
| Square     |   35.47/35.99 |   48.47/49.18 |   54.45/56.23 |     56.71/52.18 |   50.87/49.84 |
| Ours       | **3.18/3.07** | **2.14/2.06** | **1.79/1.71** | **18.12/17.78** | **2.79/2.78** |

### 2. 主结果：ADE20K

| Attack     |           FCN |         PSPNet |       DeepLabV3 |       SegFormer |      MaskFormer |
| ---------- | ------------: | -------------: | --------------: | --------------: | --------------: |
| Clean      |         33.54 |          37.68 |           39.36 |           43.74 |           45.50 |
| Random     |   22.85/22.13 |    27.72/27.36 |     25.82/24.81 |     38.02/37.64 |     25.37/24.06 |
| NES        |   24.47/23.96 |    26.57/26.26 |     23.83/23.41 |     36.35/36.06 |     34.78/34.55 |
| Bandits    |   25.10/23.67 |    25.02/23.93 |     27.52/26.36 |     36.32/35.03 |     26.14/26.91 |
| ZO-SignSGD |   23.29/22.94 |    26.82/26.47 |     25.38/24.41 |     35.22/32.18 |     33.32/32.86 |
| SignHunter |   20.15/16.72 |    24.21/20.16 |     25.40/20.48 |     32.56/28.22 |     28.78/16.78 |
| SimBA      |   24.20/21.49 |    26.36/22.92 |     25.56/22.13 |     36.70/34.81 |     35.62/33.18 |
| Square     |   23.94/22.90 |    26.87/25.89 |     27.70/26.46 |     35.43/34.76 |     26.29/26.41 |
| Ours       | **8.18/7.97** | **10.19/9.85** | **11.34/10.67** | **28.91/27.85** | **12.14/12.14** |

### 3. 搜索策略：连续扰动 vs 离散扰动

| Dataset | Model | Clean | Random | NES | Discrete |
| --- | --- | ---: | ---: | ---: | ---: |
| Cityscapes | PSPNet | 77.83 | 48.81/47.18 | 48.34/47.40 | **33.57/33.54** |
| Cityscapes | SegFormer | 80.43 | 58.59/56.07 | 58.00/55.34 | **41.70/41.70** |
| ADE20K | PSPNet | 37.68 | 26.63/26.34 | 25.67/25.52 | **23.50/23.31** |
| ADE20K | SegFormer | 43.74 | 34.72/34.45 | 34.66/34.55 | **33.68/33.53** |

### 4. DLA 设计消融

| Random | Horizontal | Vertical | Iterative | Calib Random | Calib Flip |      Cityscapes |          ADE20K |
| ------ | ---------- | -------- | --------- | ------------ | ---------- | --------------: | --------------: |
|        |            |          |           |              |            |           80.43 |           43.74 |
| ✓      |            |          |           | ✓            |            |     58.59/56.07 |     38.02/37.64 |
| ✓      |            |          |           |              | ✓          |     55.47/55.12 |     36.94/36.57 |
|        | ✓          |          |           |              | ✓          |     26.41/26.21 |     32.02/31.65 |
|        |            | ✓        |           |              | ✓          | **17.53/17.10** |     29.61/28.49 |
|        |            |          | ✓         | ✓            |            |     18.29/18.26 |     29.56/28.77 |
|        |            |          | ✓         |              | ✓          |     18.12/17.78 | **28.91/27.85** |

### 5. 极低查询预算：10 Queries

| Dataset | Clean | Random | NES | Bandits | ZO-SignSGD | SignHunter | SimBA | Square | Ours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cityscapes | 80.43 | 59.22 | 59.26 | 41.20 | 56.23 | 47.90 | 54.62 | 57.22 | **18.89** |
| ADE20K | 43.74 | 37.82 | 38.20 | 35.71 | 37.82 | 35.95 | 37.57 | 37.64 | **30.85** |

### 7.3 关键结果

- 在 Cityscapes 上，DLA 对所有模型都显著优于对比方法。
- 最亮眼的结果是 PSPNet：clean mIoU 为 77.83%，DLA 在 50 次查询下把它打到 2.14%。
- 在 200 queries 下，Cityscapes 上 DLA 的结果分别约为：FCN 3.07、PSPNet 2.06、DeepLabV3 1.71、SegFormer 17.78、MaskFormer 2.78。
- 在 ADE20K 上也保持最好结果，但整体攻击更难，说明数据集复杂度确实影响 decision-based attack 的效果。
- SegFormer 在两个数据集上都表现出相对更强的鲁棒性，是五个模型里最难攻击的一个。
- 在 10 queries 的极低预算下，DLA 依旧很强：SegFormer 在 Cityscapes 上从 80.43 降到 18.89，在 ADE20K 上从 43.74 降到 30.85。

---

## 8. 消融实验

- 使用 mIoU 作为 proxy index 比用 PAcc 更有效。
- 离散噪声优于连续随机噪声，说明“压缩搜索空间”是有效的。
- 线性噪声优于普通随机扰动，也优于更容易产生明显可见块状痕迹的 patch 类方式。
- calibration 阶段的 flip 机制优于随机更新。
- 在 ADE20K 这类尺度更复杂的数据上，horizontal / vertical 交替探索更稳健。

---

## 9. 和已有工作的关系

- 这篇论文最接近哪些方法：Boundary Attack、Sign-OPT、QEBA 等决策型黑盒攻击。
- 和已有方法相比，最大的不同：把优化目标从“越界+缩扰动”转为“mIoU 单调下降搜索”。
- 真正的新意在哪里：离散线性噪声 + 分层符号翻转校准。
- 哪些地方更像“工程改进”而不是“方法创新”：实现细节和查询预算分配策略。
- 这篇论文在整个研究脉络里的位置：分割任务 decision-based attack 的近期代表工作。

---

## 10. 我的理解（这一节不能照抄论文）

- 这篇论文真正有价值的地方，不只是做了一个新 attack，而是先把“为什么分类任务里的 decision-based attack 不能直接搬到分割里”讲清楚了。
- DLA 本质上是在做一个非常工程化但很有效的设计：先把搜索空间离散化，再把扰动结构化，最后用分层局部翻转来提高查询利用率。
- 线性噪声这个点很关键。相比 patch，它既减少局部扰动之间的覆盖冲突，也更不容易形成特别显眼的彩色块。
- 论文虽然强调 attack performance，但它的另一个意义是为后续分割鲁棒性评估提供了更强的黑盒基线。

---

## 11. 局限性

- 优化目标直接围绕 mIoU，比较适合 untargeted attack；如果要做更复杂的 targeted segmentation attack，可能还需要重新设计代理指标。
- 方法目前主要验证在常见分割模型和两个标准数据集上，面对更大规模 foundation segmentation model 时效果还不明确。
- DLA 依赖查询目标模型得到整张图的像素标签；如果现实系统只返回部分后处理结果，攻击可行性可能下降。
- 论文重点是经验效果，关于为什么“线性噪声 + flipping”在理论上更适合 segmentation，并没有特别强的理论解释。

> 可从假设过强、实验覆盖不足、开销过大、泛化不明、复现风险高等角度写。

---

## 12. 对我的启发

- 能直接借鉴的部分：代理指标单调接受准则与分层符号翻转搜索。
- 不能直接照搬的部分：`mIoU` 依赖标注可得性，任务迁移需改指标。
- 对我当前课题的启发：可将线性结构噪声推广到其他 dense prediction 攻击。
- 可以尝试的改进方向：自适应分块策略、查询预算动态分配、混合方向噪声。
- 可以作为 baseline / 对比项 / ablation 的部分：纯随机扰动、无 calibration、PAcc 代理优化。

---

## 13. 待验证问题

- 能不能把 DLA 扩展到 targeted semantic segmentation attack？
- 能不能设计比 mIoU 更适合单样本黑盒优化的 proxy index？
- 对于 SAM、Mask2Former 一类更大模型，DLA 是否依旧高效？
- 如果加入频域先验、多尺度分块或者区域重要性估计，查询效率是否还能继续提升？

---

## 14. 一句话总结

- 这篇论文的核心结论是：语义分割下的 decision-based black-box attack 不能直接照搬图像分类方法，而 DLA 通过“mIoU 代理指标 + 离散线性噪声 + 分层翻转校准”给出了一个非常强的基线。

---

## 15. 快速索引（便于二次回看）

- 核心公式：`argmax_δ Σ 1(f(x+δ)_i != y_i)`、proxy `mIoU` 优化准则。
- 核心图表：Cityscapes/ADE20K 主结果表与消融表。
- 最值得复看的章节：方法细节（Exploration + Calibration）与伪代码。
- 复现时最需要注意的点：噪声表示、分块层级、查询预算控制。

### 15.1 整合说明 / 索引

- 原始导入中的实现伪代码与细节已完整迁移到正文 `4.4`。
- `15` 节仅保留索引、自检与来源记录。

### 15.2 导入来源与完整性记录

- 主链接：https://arxiv.org/abs/2402.01220
- 本地来源：
  - `papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation.md`
  - `papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh.md`

### 15.3 已完成自检记录

- [x] 原始笔记所有内容已整理到模板中。
- [x] 图片已插入并保留在相应位置（本篇主要为文本与表格）。
- [x] 已联网补充论文信息。




## Wiki 关联

- 参考摘要：[[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation|Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- 概念锚点：[[concepts/Deep Neural Network (DNN)]]、[[concepts/Semantic Segmentation]]、[[concepts/Decision-based Black-box Attack for Segmentation]]、[[concepts/Indirect Local Attack in Segmentation]]、[[concepts/SegPGD]]、[[concepts/Proxy Index mIoU Optimization]]、[[concepts/Discrete Linear Noise]]、[[concepts/Natural Evolutionary Strategies (NES)]]、[[concepts/L-infinity Norm Ball]]、[[concepts/Query-Efficient Attack Evaluation]]
- 实体锚点：[[entities/Zhaoyu Chen]]、[[entities/Zhengyang Shan]]、[[entities/Cityscapes Dataset]]、[[entities/FCN]]、[[entities/PSPNet]]、[[entities/DeepLabv3]]、[[entities/SegFormer]]、[[entities/MaskFormer]]
- 综合页面：[[synthesis/Robust Representation and Adversarial Dynamics]]、[[synthesis/Decision-based Segmentation Attack Landscape]]、[[synthesis/Alignment Robustness Evaluation Ladder]]
