---
title: Semantic Segmentation
category: concept
tags:
  - concept
  - semantic-segmentation
  - evaluation
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh.md
  - https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html
  - https://arxiv.org/abs/1706.05587
  - https://arxiv.org/abs/2105.15203
  - https://arxiv.org/abs/2304.02643
  - https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Scene_Parsing_Through_CVPR_2017_paper.html
  - https://www.cityscapes-dataset.com/
  - https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html
  - notes/语义分割模型的Grad-CAM.md
created: 2026-04-09
updated: 2026-04-20
summary: 语义分割将每个像素映射到语义类别，是场景理解中的密集预测基础任务，主评估指标通常是 mIoU。
provenance:
  extracted: 0.72
  inferred: 0.25
  ambiguous: 0.03
---

# Semantic Segmentation

## 定义

语义分割是像素级分类任务：为图像中每个像素分配一个语义类别（如 road、person、sky）。

## 任务边界

- 与图像分类不同：输出不是单个标签，而是整张像素标签图。
- 与目标检测不同：不只给框，而是给每个像素类别。
- 与实例分割不同：语义分割不区分同类不同实例。
- 与全景分割关系：全景分割可视为把语义分割与实例分割统一到同一输出。^[inferred]

## 常用评估

- mIoU（mean Intersection-over-Union）是最常用主指标。
- 在黑盒攻击语境中，常直接比较“固定 query budget 下的 mIoU 下降幅度”。

## 方法演进（简）

- FCN（CVPR 2015）把分类网络改造为端到端 `pixels-to-pixels` 的全卷积分割框架，并使用 skip architecture。
- DeepLabv3（2017）系统化使用 atrous convolution 与 ASPP 处理多尺度上下文。
- SegFormer（2021）采用分层 Transformer 编码器 + 轻量 MLP decoder，强调无需位置编码。
- SAM（2023）将分割推进到 promptable 基础模型范式，并发布 SA-1B 数据规模。^[inferred]

## 典型数据集

- Cityscapes：城市街景分割基准，官方强调 pixel-level / instance-level / panoptic 多任务评测。
- ADE20K：场景解析基准，CVPR 2017 论文基准设置为 150 个 object/stuff 类别。

## 在当前库中的作用

- 这是 [[concepts/Decision-based Black-box Attack for Segmentation]] 的上位任务概念。
- 在本库攻击评估里，语义分割通常通过 mIoU 与 query efficiency 共同刻画鲁棒性。
- 在解释性分析里，[[concepts/Grad-CAM for Semantic Segmentation]] 提供了一种把像素级 logit map 压成标量目标、再回投到中间层的可视化路径，可用于比较 `clean / adv / diff` 的层级响应。^[inferred]

## 联网补充

- 2015-06-08（CVPR 2015）：FCN 将语义分割确立为可端到端训练的密集预测范式，并报告 VOC2012 `62.2% mean IU`。
- 2017-06-17（arXiv）：DeepLabv3 明确以 atrous convolution + ASPP 作为多尺度上下文建模核心。
- 2021-05-31（arXiv）：SegFormer 把 Transformer 分割结构做成高效统一框架，并强调“无位置编码”。
- 2023-04-05（arXiv）：Segment Anything 提出可提示（promptable）分割模型与 SA-1B，推动了通用分割基础模型路线。
- Cityscapes 官方首页当前仍将语义分割列为核心 benchmark 任务之一（并与 instance/panoptic 并列）。

## 关联

- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Grad-CAM for Semantic Segmentation]]
- [[concepts/Proxy Index mIoU Optimization]]
- [[concepts/Query-Efficient Attack Evaluation]]
- [[notes/语义分割模型的Grad-CAM|语义分割模型的 Grad-CAM]]
- [[entities/Cityscapes Dataset]]
- [[entities/ADE20K Dataset]]
- [[entities/FCN]]
- [[entities/PSPNet]]
- [[entities/DeepLabv3]]
- [[entities/SegFormer]]
- [[entities/MaskFormer]]
- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
