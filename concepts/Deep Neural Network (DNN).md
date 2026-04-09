---
title: Deep Neural Network (DNN)
category: concept
tags:
  - concept
  - architecture
  - adversarial-attack
  - semantic-segmentation
  - robustness
sources:
  - papers_sources/Delving into Decision-based Black-box Attacks on Semantic Segmentation/Delving into Decision-based Black-box Attacks on Semantic Segmentation_zh.md
created: 2026-04-09
updated: 2026-04-09
summary: DNN 通过多层可学习表示进行端到端建模，是现代语义分割与对抗鲁棒性研究的基础模型范式。
provenance:
  extracted: 0.76
  inferred: 0.20
  ambiguous: 0.04
---

# Deep Neural Network (DNN)

## 定义

DNN 是由多层参数化非线性变换组成的神经网络，通过端到端训练学习从输入到任务输出的分层表示。

## 在语义分割中的角色

- 语义分割主流方法（FCN、DeepLab、PSPNet、SegFormer）本质上都建立在 DNN 表示学习能力之上。
- 在 black-box attack 语境下，攻击目标是 DNN 的像素级决策输出，而非单标签分类输出。

## 联网补充

- Nature 2015 的综述将 deep learning 定义为“多层表示学习”，核心是通过多级表示逐层抽取更抽象特征。
- AlexNet（NeurIPS 2012）证明了深层卷积网络在大规模视觉识别上的显著性能跃迁，推动 DNN 成为视觉主流范式。
- FCN（CVPR 2015）把卷积网络扩展为 end-to-end、pixels-to-pixels 的密集预测框架，奠定了 DNN 在语义分割中的基础路径。
- Explaining and Harnessing Adversarial Examples（2014/2015）显示小幅最坏方向扰动可使 DNN 高置信误判，直接引出鲁棒性与对抗攻击研究。

## 关联

- [[references/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[notes/Delving into Decision-based Black-box Attacks on Semantic Segmentation]]
- [[concepts/Decision-based Black-box Attack for Segmentation]]
- [[concepts/Proxy Index mIoU Optimization]]
- [[concepts/Query-Efficient Attack Evaluation]]

