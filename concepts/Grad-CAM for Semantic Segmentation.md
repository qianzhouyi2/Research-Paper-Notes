---
title: Grad-CAM for Semantic Segmentation
category: concept
tags:
  - concept
  - semantic-segmentation
  - interpretability
  - diagnostics
sources:
  - notes/语义分割模型的Grad-CAM.md
created: 2026-04-20
updated: 2026-04-20
summary: 将 Grad-CAM 从图像分类迁移到语义分割时，关键是把像素级 logits 构造成可求梯度的标量目标，再回投到中间层形成空间热图。
provenance:
  extracted: 0.74
  inferred: 0.23
  ambiguous: 0.03
---

# Grad-CAM for Semantic Segmentation

## 定义

Grad-CAM 在语义分割里的核心任务不是“解释单个类别分数”，而是“先从密集 logits map 中构造一个标量目标，再解释这个目标依赖了中间层哪些空间位置”。这使它能回答“模型把某一类判出来时主要依赖了哪里”，而不是只回答整张图属于哪一类。^[extracted]

## 与分类版 CAM 的核心差异

- 分类版 CAM 默认面对的是单个类别分数；语义分割输出则是 `logits ∈ R^{B x C_cls x H x W}` 的像素级分数图。^[extracted]
- 因此在分割场景里，必须先围绕类别 `c` 定义一个标量 `target_score`，否则无法直接对中间特征图生成一张热图。^[extracted]
- 这一步决定了解释对象到底是“某类像素区域的支持证据”还是“整张图上的平均倾向”。^[inferred]

## 一个实用构造

- 先取目标类别的 logit map：`class_logits = logits[:, c]`。^[extracted]
- 再根据当前预测构造 `target_mask = prediction == c`。^[extracted]
- 若该类像素存在，就对该区域的 logits 求平均；若不存在，则退化为整张 logit map 的平均，避免目标分数为空。^[extracted]
- 然后对选定中间层 `feature_map` 求梯度，在空间维上做平均得到通道权重，再计算 `ReLU(sum(weights * feature_map))` 并上采样回输入大小。^[extracted]

```text
class_logits = logits[:, c]
prediction = logits.argmax(dim=1)
target_mask = prediction == c
target_score = mean(class_logits[target_mask]) or mean(class_logits)
gradients = grad(target_score, feature_map)
weights = spatial_mean(gradients)
cam = relu(sum(weights * feature_map over channel))
cam = resize_to_input_resolution(cam)
```

## 解释边界

- 这类热图解释的是“哪些位置在支持当前 `target_score`”，不是严格的因果证明。^[extracted]
- 结果会明显依赖目标分数定义、目标层选择、mask 构造方式与插值策略。^[inferred]
- 把 `clean / adv / diff` 按层并排对照，更适合观察表征漂移和脆弱点迁移，而不是单独证明攻击机理。^[inferred]

## 在当前库中的作用

- 它为 [[concepts/Semantic Segmentation]] 补上了一个可解释性视角。
- 它可作为 [[concepts/Adversarial Vulnerability Profiling for Segmentation Models]] 的诊断工具，用来比较干净输入与扰动输入在不同层上的响应差异。^[inferred]
- 当前对应的长笔记是 [[notes/语义分割模型的Grad-CAM|语义分割模型的 Grad-CAM]]。

## 关联

- [[concepts/Semantic Segmentation]]
- [[concepts/Adversarial Vulnerability Profiling for Segmentation Models]]
- [[notes/语义分割模型的Grad-CAM|语义分割模型的 Grad-CAM]]
