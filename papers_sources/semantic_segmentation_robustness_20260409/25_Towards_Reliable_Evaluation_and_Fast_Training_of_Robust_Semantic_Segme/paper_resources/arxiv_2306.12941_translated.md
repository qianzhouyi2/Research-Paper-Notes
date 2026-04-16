# Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models

- Authors: Francesco Croce, Naman D. Singh, Matthias Hein
- Affiliations: EPFL; University of Tübingen; Tübingen AI Center
- Notes: `*` equal contribution; correspondence: `naman-deep.singh@uni-tuebingen.de`
- Source HTML: `arxiv_2306.12941_translated.html`

## Abstract

Adversarial robustness has been studied extensively in image classification, especially for the ℓ_(∞)-threat model, but significantly less so for related tasks such as object detection and semantic segmentation, where attacks turn out to be a much harder optimization problem than for image classification. We propose several problem-specific novel attacks minimizing different metrics in accuracy and mIoU. The ensemble of our attacks, SEA, shows that existing attacks severely overestimate the robustness of semantic segmentation models. Surprisingly, existing attempts of adversarial training for semantic segmentation models turn out to be weak or even completely non-robust. We investigate why previous adaptations of adversarial training to semantic segmentation failed and show how recently proposed robust ImageNet backbones can be used to obtain adversarially robust semantic segmentation models with up to six times less training time for Pascal-Voc and the more challenging Ade20K. The associated code and robust models are available at https://github.com/nmndeep/robust-segmentation.

## Keywords

Semantic segmentation Adversarial attacks Robust models


## 1 Introduction

The vulnerability of neural networks to adversarial perturbations, that is small changes of the input that can drastically modify the output of the models, has been extensively studied [6, 41, 20, 24], in particular for image classification. Adversarial attacks have been developed for various threat models, including ℓ_(p)-bounded perturbations [8, 9, 35], sparse attacks [7, 13], and those defined by perceptual metrics [44, 27]. At the same time, evaluating adversarial robustness in semantic segmentation, undoubtedly an important vision task, has received much less attention. While a few early works [47, 31, 2] have proposed adversarial attacks in different threat models, [21, 1] have recently shown that, even for ℓ_(∞)-bounded pertubations, significant improvements are possible over the standard PGD attack [30] based on the sum of pixel-wise cross-entropy losses. In fact, the key difference of semantic segmentation to image classification is that for the former one has to flip the predictions of all pixels, not just the prediction for the image, which is a much harder optimization problem. This might also explain why only few works [48, 21] have produced robust semantic segmentation models via variants of adversarial training [30]. In this work, we address both the evaluation and training of adversarially robust semantic segmentation models.

Challenges of adversarial attacks in semantic segmentation. We analyze why the cross-entropy loss, in contrast to image classification, is not a suitable objective for generating strong adversarial attacks against semantic segmentation models. To address this issue, we propose novel loss functions that are specifically designed for semantic segmentation (Sec. 3.4), as they aim at flipping the decisions of all pixels simultaneously. Moreover, in contrast to prior work we do not only attack accuracy but also mIoU (mean intersection over union), the most common performance metric in semantic segmentation. As the direct optimization of mIoU is intractable since it depends on the outcome on all test images together, we derive an upper bound on mIoU in terms of an imagewise loss which can be used as attack objective (Sec. 3.2). Finally, we study several improvements for the PGD-attack (Sec. 3.5) which boost performance.

- Figure 1 panel layout: `ground truth` | `clean model: original` | `clean model: target grass` | `clean model: target sky` | `robust model: original` | `robust model: target grass` | `robust model: target sky`.

Figure 1: Effect of adversarial attacks on semantic segmentation models. For a validation image of Ade20K (first column, with ground truth mask), we show the image perturbed by targeted ℓ_(∞)-attacks (ϵ_(∞) = 2/255, target class “grass” or “sky”), and the predicted segmentation. For a clean model the attack completely alters the segmentation map, while our robust model (UPerNet + ConvNeXt-T trained with 5-step PIR-AT for 128 epochs) is minimally affected. For illustration, we use targeted attacks, and not untargeted ones as in the rest of the paper. More illustrations in Appendix 0.D.

Strong evaluation of robustness with SEA. In Sec. 3.6, we introduce SEA (Semantic Ensemble Attack), a reliable robustness evaluation for semantic segmentation for the ℓ_(∞)-threat model via an ensemble of our three complementary attacks, two designed to minimize average pixel accuracy and one targeting mIoU. In the experiments, we show that recent SOTA attacks, SegPGD [21] and CosPGD [1], may significantly overestimate the robustness of semantic segmentation models, in particular regarding mIoU. Tab. 1 illustrates, for multiple robust and non-robust models, that our individual novel attacks and in particular our ensemble SEA consistently outperform SOTA attacks, with improvements of more than 17.2% in accuracy and 10.3% in mIoU.

Robust semantic segmentation models with PIR-AT. It is interesting to note that no adversarially robust semantic segmentation models with respect to ℓ_(∞)-threat model are publicly available. The models of DDC-AT [48] turned out to be non-robust, see Tab. 1. This implies that adversarial training (AT) is much harder for semantic segmentation than for image classification, likely due to the much more difficult attack problem. In fact, to obtain satisfactory robustness with AT we had to increase the number of epochs compared to clean training and use many (up to 5) attack steps. However, this yields a high computational cost, that is prohibitive for scaling up to large architectures. We drastically reduce this cost by leveraging recent progress in robust ImageNet classifiers [16, 39, 28]. In fact, we introduce Pre-trained ImageNet Robust AT (PIR-AT), where we initialize the backbone of the segmentation model with an ℓ_(∞)-robust ImageNet classifier. This allows us to i) reduce the cost of AT (robust ImageNet classifier are widely available e.g. in RobustBench [12]), and ii) achieve SOTA robustness on segmentation datasets. On Pascal-Voc [18], Tab. 1 shows that our approach PIR-AT (see Sec. 4 for details) attains 71.7% robust average pixel accuracy at attack radius ϵ_(∞) = 8/255 compared to 0.0% of DDC-AT, with negligible drop in clean performance compared to clean training (92.7% vs. 93.4% accuracy, 75.9% vs 77.2% mIoU). For the more challenging Ade20K [50] we obtain the first, to our knowledge, robust models, with up to 55.5% robust accuracy at ϵ_(∞) = 4/255. Finally, we show that PIR-AT consistently outperforms AT across segementation networks (PSPNet [49], UPerNet [46], Segmenter [40]).

## 2 Related Work

Adversarial attacks for semantic segmentation. ℓ_(∞)-bounded attacks on segmentation models have been first proposed by [31], which focuses on targeted (universal) attacks, and [2], using FGSM [19] or PGD on the cross-entropy loss. Recently, [21, 1] revisited the loss used in the attack to improve the effectiveness of ℓ_(∞)-bounded attacks, and are closest in spirit to our work (see extended discussion in Sec. 0.C.4). Additionally, there exist attacks for other threat models, including unconstrained, universal and patch attacks [47, 11, 32, 38, 25, 33, 36]. In particular, [36] introduces an algorithm to minimize the ℓ_(∞)-norm of the perturbations such that a fixed percentage of pixels is successfully attacked. While their threat model is quite different, we adapt our attacks to it and provide a comparison in Sec. 0.C.3: especially for robust models, our attacks outperform the method of [36] by large margin.

Robust segmentation models. Only a few works have developed defenses for semantic segmentation models. [45] proposes a method to detect attacks, while stating that adversarial training is hard to adapt for semantic segmentation. Later, DDC-AT [48] attempts to integrate adversarial points during training exploiting additional branches of the networks. The seemingly robust DDC-AT has been shown to be non-robust using SegPGD in [21] at ϵ_(∞) = 8/255, whereas we show with SEA (Tab. 1) that it is non-robust even for ϵ_(∞) = 2/255 where SegPGD still flags robustness. Finally, [10, 26] present defenses based on denoising the input, with Autoencoders or Wiener filters, to remove perturbations before feeding it to clean models. These methods are only tested via attacks with limited budgets, and similar techniques to protect image classifiers have been rendered ineffective with stronger evaluation [3, 42].

## 3 Adversarial Attacks for Semantic Segmentation

First, we discuss the general setup and how to design loss functions for attacks which target either mIoU or accuracy of semantic segmentation models. Then, we introduce our three novel objectives and discuss how to improve the optimization scheme of PGD in this context. Finally, we present our attack ensemble SEA (Sec. 3.6) for reliable evaluation of adversarial robustness including our three proposed diverse attacks, which significantly improves over existing SOTA attacks.

### 3.1 Setup

The goal of semantic segmentation consists in classifying each pixel of a given image into the K available classes (corresponding to different objects or background). We denote by f : ℝ^(w × h × c) → ℝ^(w × h × K) a segmentation model which for an image x of size w × h (and c color channels) returns u = f(x), where u_(ij) ∈ ℝ^(K) contains the logit of each of the K classes for the pixel x_(ij). The class predicted by f for x_(ij) is given by m_(ij) = arg max _(k = 1, …, K)u_(ijk), and m ∈ ℝ^(w × h) is the segmentation map of x. Given the ground truth map y ∈ ℝ^(w × h), the average pixel accuracy of f for x is $\frac{1}{w \cdot h}{\sum_{i,j}{{\mathbb{I}}{({{\mathbf{m}}_{ij} = {\mathbf{y}}_{ij}})}}}$. In the following, we use index a for the pixel (i,j) to simplify the notation. The goal of an adversarial attack on f is to change the input x such that as many pixels as possible are mis-classified. This can be formalized as the optimization problem

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -- -----
     ${{{{\max\limits_{\mathbf{δ}}{\sum\limits_{a}{\mathcal{L}{({f{({{\mathbf{x}} + {\mathbf{δ}}})}_{a}},{\mathbf{y}}_{a})}}}}\quad\text{s. th.}\quad\left\| {\mathbf{δ}} \right\|_{\infty}} \leq \epsilon},{{{\mathbf{x}} + {\mathbf{δ}}} \in {\lbrack 0,1\rbrack}^{{wh} \times c}}},$      (1)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -- -----

where we use the ℓ_(∞)-threat model for the perturbations δ, x + δ is restricted to be an image, and ℒ : ℝ^(K) × ℝ → ℝ is a differentiable loss whose maximization induces mis-classification. This can then be (approximately) solved by techniques for constrained optimization such as projected gradient descent (PGD) as suggested for image classification in [30].

Background pixels. In semantic segmentation, it is common to exclude pixels that belong to the background class when training or computing the test performance. However, it is unrealistic for an attacker to modify only non-background pixels. Thus semantic segmentation models must be robust for all pixels regardless of how they are classified or what their ground-truth label is. Therefore, we train all our models with an additional background class. This has little impact on segmentation performance, see Sec. 0.C.2, and yields a realistic and practically relevant definition of adversarial robustness.

### 3.2 How to efficiently attack mIoU

In semantic segmentation it is common to use the Intersection over Union (IoU) as performance metric, averaged over classes (mIoU). The mIoU is typically computed across all the images in the test set and not as average over the images. As this makes image-wise optimization infeasible, a direct optimization of mIoU is intractable. We derive in the following an upper bound on mIoU which can be efficiently optimized and which we use as an objective for an attack on mIoU.

Let TP_(s) and FP_(s) be true and false positive pixels of class s (and accordingly true and false negatives) for all (test) images. For each class s the IoU is given as

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------- -- -----
     ${\text{IoU}_{s} = \frac{TP_{s}}{{TP_{s}} + {FP_{s}} + {FN_{s}}} \leq \frac{TP_{s}}{{TP_{s}} + {FN_{s}}} = \frac{TP_{s}}{N_{s}} = \text{Acc}_{s}},$      (2)
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------- -- -----

where Acc_(s) is the accuracy of the s-th class and N_(s) = TP_(s) + FN_(s) the total number of pixels of class s (in the test set). Thus we get

  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----
     ${\text{mIoU} = {\frac{1}{K}{\sum\limits_{s = 1}^{K}{IoU}_{s}}} \leq {\frac{1}{K}{\sum\limits_{s = 1}^{K}{Acc}_{s}}} = {\frac{1}{K}{\sum\limits_{s = 1}^{K}\frac{TP_{s}}{N_{s}}}} = {\frac{1}{K}{\sum\limits_{i = 1}^{I}{\sum\limits_{s = 1}^{K}{\frac{1}{N_{s}}TP_{si}}}}}},$      (3)
  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----

where I is the total number of images in the test set and TP_(si) denotes the number of true positives of class s in image i (we use ${TP_{s}} = {\sum_{i = 1}^{I}{TP_{si}}}$). Thus mIoU is upper-bounded by class-balanced accuracy. We can interpret the last expression as an image-wise weighted accuracy, where correct pixels of class s have an image-independent weight of $\frac{1}{N_{s}}$. This is an image-wise loss which we can optimize as in Eq. (1). In practice, to not use any information from the test set, we obtain the number of pixels per class N_(s) from the training set. We use this upper bound as objective for our mIoU-specific attack in Sec. 3.4.

| Model                                                | Clean (Acc / mIoU) | ϵ_(∞)    | CosPGD Acc | CosPGD mIoU | SegPGD Acc | SegPGD mIoU | ℒ_(JS) Acc | ℒ_(JS) mIoU | ℒ_(MCE) Acc | ℒ_(MCE) mIoU | ℒ_(MCE-Bal) Acc | ℒ_(MCE-Bal) mIoU | SEA Acc | SEA mIoU |
| ---------------------------------------------------- | ------------------ | -------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ----------- | ------------ | --------------- | ---------------- | ------- | -------- |
| Pascal-Voc: UPerNet+ConvNeXt-T (clean training)      | 93.4 / 77.2        | 0.25/255 | 76.6       | 48.0        | 74.0       | 43.7        | 72.4       | 41.9        | 71.2        | 39.6         | 76.6            | 40.3             | 70.0    | 36.9     |
| Pascal-Voc: UPerNet+ConvNeXt-T (clean training)      | 93.4 / 77.2        | 0.5/255  | 46.9       | 24.0        | 43.0       | 18.6        | 36.7       | 16.3        | 33.2        | 13.0         | 37.9            | 10.1             | 31.1    | 8.6      |
| Pascal-Voc: UPerNet+ConvNeXt-T (clean training)      | 93.4 / 77.2        | 1/255    | 17.2       | 8.1         | 12.9       | 4.2         | 8.2        | 3.4         | 5.9         | 1.6          | 6.9             | 0.8              | 4.9     | 0.6      |
| Pascal-Voc: PSPNet-50 (DDC-AT [48])                  | 92.8 / 76.0        | 2/255    | 7.1        | 3.9         | 3.9        | 2.2         | 1.2        | 0.8         | 0.5         | 0.3          | 2.1             | 1.0              | 0.2     | 0.1      |
| Pascal-Voc: PSPNet-50 (DDC-AT [48])                  | 92.8 / 76.0        | 4/255    | 6.5        | 3.5         | 2.6        | 1.6         | 0.3        | 0.1         | 0.2         | 0.1          | 0.5             | 0.0              | 0.1     | 0.0      |
| Pascal-Voc: PSPNet-50 (DDC-AT [48])                  | 92.8 / 76.0        | 8/255    | 4.5        | 3.2         | 2.1        | 1.3         | 0.0        | 0.0         | 0.0         | 0.0          | 0.0             | 0.0              | 0.0     | 0.0      |
| Pascal-Voc: UPerNet+ConvNeXt-T (PIR-AT ours, Tab. 2) | 92.7 / 75.9        | 4/255    | 89.0       | 65.6        | 88.7       | 64.8        | 88.7       | 64.9        | 89.2        | 65.9         | 90.4            | 67.4             | 88.6    | 64.9     |
| Pascal-Voc: UPerNet+ConvNeXt-T (PIR-AT ours, Tab. 2) | 92.7 / 75.9        | 8/255    | 77.8       | 47.3        | 74.2       | 41.4        | 73.9       | 41.3        | 74.0        | 40.6         | 77.4            | 38.4             | 71.7    | 34.6     |
| Pascal-Voc: UPerNet+ConvNeXt-T (PIR-AT ours, Tab. 2) | 92.7 / 75.9        | 12/255   | 56.6       | 26.4        | 45.3       | 15.8        | 38.6       | 15.1        | 31.5        | 10.3         | 36.9            | 6.7              | 28.1    | 5.5      |
| Ade20K: UPerNet + ConvNeXt-T (AT 2 steps, Tab. 3)    | 73.4 / 36.4        | 0.25/255 | 61.3       | 24.6        | 60.9       | 24.4        | 58.8       | 23.1        | 59.4        | 23.8         | 61.5            | 21.6             | 58.5    | 20.9     |
| Ade20K: UPerNet + ConvNeXt-T (AT 2 steps, Tab. 3)    | 73.4 / 36.4        | 0.5/255  | 46.5       | 14.6        | 41.1       | 12.2        | 29.8       | 7.1         | 28.5        | 6.3          | 33.1            | 5.9              | 27.5    | 5.1      |
| Ade20K: UPerNet + ConvNeXt-T (AT 2 steps, Tab. 3)    | 73.4 / 36.4        | 1/255    | 18.3       | 4.4         | 9.9        | 2.2         | 1.8        | 0.3         | 1.1         | 0.4          | 1.6             | 0.1              | 0.8     | 0.0      |
| Ade20K: UPerNet + ConvNeXt-T (PIR-AT ours, Tab. 2)   | 70.5 / 31.7        | 4/255    | 57.5       | 19.9        | 55.9       | 19.0        | 55.9       | 18.9        | 56.8        | 20.0         | 58.2            | 17.9             | 55.5    | 17.2     |
| Ade20K: UPerNet + ConvNeXt-T (PIR-AT ours, Tab. 2)   | 70.5 / 31.7        | 8/255    | 37.6       | 9.9         | 28.5       | 7.4         | 28.5       | 7.2         | 28.5        | 6.6          | 31.1            | 5.3              | 26.4    | 4.9      |
| Ade20K: UPerNet + ConvNeXt-T (PIR-AT ours, Tab. 2)   | 70.5 / 31.7        | 12/255   | 19.5       | 4.1         | 5.5        | 1.2         | 5.2        | 1.1         | 3.7         | 0.9          | 5.2             | 0.9              | 3.1     | 0.4      |
| Ade20K: Segmenter + ViT-S (PIR-AT ours, Tab. 2)      | 69.1 / 28.7        | 4/255    | 57.4       | 17.5        | 57.3       | 17.5        | 55.6       | 16.6        | 56.9        | 17.8         | 57.6            | 15.6             | 55.3    | 14.9     |
| Ade20K: Segmenter + ViT-S (PIR-AT ours, Tab. 2)      | 69.1 / 28.7        | 8/255    | 41.7       | 9.7         | 38.5       | 8.6         | 34.2       | 7.7         | 36.2        | 8.5          | 37.8            | 5.6              | 33.3    | 5.4      |
| Ade20K: Segmenter + ViT-S (PIR-AT ours, Tab. 2)      | 69.1 / 28.7        | 12/255   | 25.6       | 4.9         | 17.4       | 2.9         | 11.2       | 2.2         | 10.5        | 2.2          | 11.7            | 1.3              | 8.9     | 1.1      |

Table 1: Comparison of attacks. We compare the performance of all attacks for a budget of 300 iterations for clean and robust models trained on Pascal-Voc and Ade20K. We report average pixel accuracy and mIoU (clean performance is next to the model name). Our JS and MCE attacks outperform SegPGD in almost all cases, whereas MCE-Bal targeted on minimizing mIoU achieves in most cases the best mIoU. Our ensemble attack SEA outperforms both CosPGD [1] and SegPGD [21] by large margin. For each metric: best attack in bold, second best underlined.

### 3.3 Why do attacks on semantic segmentation require new loss functions compared to image segmentation?

The main challenge of an attack on semantic segmentation is to flip the decisions of many pixels ( ≥ 10⁵) of an image at the same time, ideally with only a few iterations. To tackle this problem it is therefore illustrative to consider the gradient of the total loss with respect to the input. We denote by u ∈ ℝ^(wh × K) the logit output for the full image and by u_(ak) the logit of class k ∈ {1, …, K} for pixel a, which yields

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----
     ${{\nabla_{\mathbf{x}}{\sum\limits_{a}{\mathcal{L}{(u_{a},y_{a})}}}} = {\sum\limits_{a}{\sum\limits_{k = 1}^{K}{\frac{\partial\mathcal{L}}{\partial u_{ak}}{(u_{a},y_{a})}{\nabla_{\mathbf{x}}u_{ak}}}}}}.$      (4)
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----

The term $\sum_{k = 1}^{K}{\frac{\partial\mathcal{L}}{\partial u_{ak}}{(u_{a},y_{a})}}$ can be interpreted as the influence of pixel a. The main problem is that successfully attacked pixels (i.e. pixels wrongly classified) typically have non-zero gradients, and in some cases, e.g. for the cross-entropy loss, these gradients have the largest magnitude for mis-classified pixels (see next paragraph). Thus already misclassified pixels have strong influence on the iterate updates, and in the worst case prevent that the decisions of still correctly classified pixels are flipped.

The losses we introduce in the following have their own strategy to cope with this problem and yield either implicitly or explicitly a different weighting of the contributions of each pixel. Some losses are easier described in terms of the predicted probability distribution p ∈ ℝ^(K) via the softmax function: ${\mathbf{p}}_{r} = {e^{{\mathbf{u}}_{r}}/{\sum_{k = 1}^{K}e^{{\mathbf{u}}_{k}}}}$, k = 1, …, K (p is a function of u). Moreover, we omit the pixel-index a in the following for easier presentation.

#### 3.3.1 Why does the cross-entropy loss not work for semantic segmentation?

The most common choice as objective function in PGD based attacks for image classification is the cross-entropy loss, i.e. ${\mathcal{L}_{\text{CE}}{({\mathbf{u}},y)}} = {- {\log{\mathbf{p}}_{y}}} = {{- {\mathbf{u}}_{y}} + {\log\left( {\sum_{j = 1}^{K}e^{{\mathbf{u}}_{j}}} \right)}}$. Maximizing CE-loss is equivalent to minimizing the predicted probability of the correct class. The cross-entropy loss is unbounded, which is problematic for semantic segmentation as misclassified pixels are still optimized instead of focusing on correctly classified pixels. It holds

  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${\frac{\partial{\mathcal{L}_{\text{CE}}{({\mathbf{p}},{\mathbf{e}}_{y})}}}{\partial{\mathbf{u}}_{k}} = {{- \delta_{yk}} + {{\mathbf{p}}_{k}\text{~and~}\frac{K}{K - 1}{({1 - {\mathbf{p}}_{y}})}^{2}}} \leq \left\| {\nabla_{\mathbf{u}}\mathcal{L}_{\text{CE}}} \right\|_{2}^{2} \leq {{{({1 - {\mathbf{p}}_{y}})}^{2} + 1} - {\mathbf{p}}_{y}}}.$$
  -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

The bounds on the gradient norm are monotonically increasing as p_(y) → 0 (see Appendix 0.A). Therefore maximally misclassified pixels have the strongest influence in Eq. (4) for the CE-loss which explains why it does not work well as an objective for attacks on semantic segmentation, as shown in [1, 21].

### 3.4 Novel attacks on semantic segmentation

In the following we introduce our novel attacks, which address the shortcomings of the cross-entropy loss. All of them are specifically designed for semantic segmentation and have complementary properties.

Jensen-Shannon (JS) divergence:

    the main problem of the cross-entropy loss is that the norm of the gradient is increasing as p_(y) → 0, that is the more the pixel is mis-classified. We propose to use instead the Jensen-Shannon divergence, a loss which has not been used for adversarial attacks and has properties which make it particularly useful for attacks on semantic segmentation. Given two distributions q₁ and q₂, the Jensen-Shannon divergence is defined as

      -- --------------------------------------------------------------------- --
         D_(JS)(q₁∥q₂) = (D_(KL)(q₁∥m)+D_(KL)(q₂∥m))/2, with  m = (q₁+q₂)/2,
      -- --------------------------------------------------------------------- --

    where D_(KL) indicates the Kullback–Leibler divergence. Let e_(y) be the one-hot encoding of the target y. We set ℒ_(JS)(u,y) = D_(JS)(p∥e_(y)). As D_(JS) measures the similarity between two distributions, maximizing ℒ_(JS) drives the prediction of the model away from the ground truth e_(y). Unlike the CE loss, the JS divergence is bounded, and thus the influence of every pixel is limited. The most relevant property is that the gradient of ℒ_(JS) vanishes as p_(y) → 0 (proof in Appendix 0.A )

      -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --
         $${{{\lim\limits_{{\mathbf{p}}_{y}\rightarrow 0}\frac{\partial{\mathcal{L}_{\text{JS}}{({\mathbf{u}},y)}}}{\partial{\mathbf{u}}_{k}}} = {0,\text{for}}}\quad{k = {1,\ldots,K}}}.$$
      -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --

    Thus ℒ_(JS) automatically down-weights contributions from mis-classified pixels and in turn pixels which are still correctly classified get a higher weight in the gradient in Eq (4). Note that this property is completely undesired for a loss in image classification as this implies that maximally wrong predictions do not lead to gradients, making it very difficult to change very confident wrong predictions. However, for attacks on semantic segmentation this conveniently allows us to simultaneously optimize all pixels, without requiring any masking of the misclassified ones as for other attacks (see next paragraph). Our ℒ_(JS)-based attack is in 7 out of the 18 cases considered in Tab. 1 the best single attack. In particular, in average pixel accuracy it always outperforms CosPGD [1], and SegPGD [21] in 15 out of 18 cases and is otherwise equal.

Masked cross-entropy (unbalanced and balanced):

    we use the masked cross-entropy (MCE) for two attacks, targeting either standard accuracy or mIoU. The main idea in order to avoid over-optimizing mis-classified pixels is to apply a mask to the CE loss which excludes misclassified pixels from the loss: this gives

      -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
         $${{\mathcal{L}_{\text{MCE}}{({\mathbf{u}},y)}} = {{{{\mathbb{I}}{({{\underset{j = {1,\ldots,K}}{\arg\max}{\mathbf{u}}_{j}} = y})}} \cdot \mathcal{L}_{\text{CE}}}{({\mathbf{u}},y)}}},$$
      -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

    the unbalanced loss addressing average pixel accuracy. To minimize the upper bound on mIoU in Eq. (3) we instead use the balanced masked cross-entropy loss

      -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
         $${{\mathcal{L}_{\text{MCE-Bal}}{({\mathbf{u}},y)}} = {{{\frac{1}{N_{y}}{\mathbb{I}}{({{\underset{j = {1,\ldots,K}}{\arg\max}{\mathbf{u}}_{j}} = y})}} \cdot \mathcal{L}_{\text{CE}}}{({\mathbf{u}},y)}}},$$
      -- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

    where we weight the loss of each pixel according to the number of pixels N_(y) of the ground-truth class in the training images. The downside of a mask is that the loss becomes discontinuous and ignoring mis-classified pixels can lead to changes which revert back mis-classified pixels into correctly classified ones with the danger of oscillations. We note that [47] used masking for a margin based loss and [31] for targeted attacks to not optimize pixels already classified into the target class with confidence higher than a fixed threshold. However, the MCE-loss has not been explored for ℓ_(∞)-bounded untargeted attacks, and turns out to be the best among the five single attacks for average pixel accuracy in 11 out of 18 cases, with larger margins for the larger radii, see Tab. 1. Our MCE-Bal loss targeting mIoU achieves in 15 out of 18 cases the best mIoU of all five single attacks.

#### 3.4.1 Performance of our single attacks.

In Tab. 1 we compare the performance of our three novel attacks, i.e. the JS, MCE, and MCE-Bal losses optimized using APGD [14] with the scheme detailed in Sec. 3.5, to CosPGD [1] and SegPGD [21] on various robust and non-robust models trained on Pascal-Voc and Ade20K. All attacks have the same budget of 300 iterations. Regarding average pixel accuracy our attacks are better than the SOTA attacks in 15 out of 18 cases and equal otherwise. Regarding mIoU our attacks are better in 17 out of 18 cases and are only 0.1% worse in the remaining one. This highlights that our novel attacks outperform CosPGD and SegPGD, and at the same time are complementary in where they perform best. This motivates our ensemble attack SEA in Sec. 3.6.

### 3.5 Optimization techniques for adversarial attacks on semantic segmentation

We discuss next how to improve the optimization scheme of the attacks in semantic segmentation, in particular to obtain more efficient algorithms. For optimizing the problem in Eq. (1) we use APGD [14], since it has been shown to outperform the original PGD [30]. While it was designed for image classification, it can be applied to general objectives and constraint sets.

Progressive radius reduction. We noticed in early experiments that, when used in the standard formulation, the optimization may at times get stuck, regardless of the objective function. At the same time, increasing the radius, i.e. larger ϵ_(∞), reduces robust accuracy eventually to zero meaning the gradient information is valid (there is no gradient masking). In order to mitigate this problem and profit from the result of larger ϵ_(∞), we run the attack starting with a larger radius and then use its projection onto the feasible set as starting point for the attack with a smaller radius, similar to [15] for ℓ₁-attacks. We split the budget of iterations into three slots (with ratio 3 : 3 : 4 as in [15]) with attack radii 2 ⋅ ϵ_(∞), 1.5 ⋅ ϵ_(∞) and ϵ_(∞) respectively.

[Refer to caption]

Figure 2: Comparison of const-ϵ and red-ϵ optimization schemes. Attack accuracy for the robust PIR-AT UPerNet+ConvNeXt-T model from Table 1 on Pascal-Voc, across different losses for the same iteration budget. The radius reduction (red-ϵ) scheme performs best across all attacks, and it even improves the worst-case over all attacks.

Radius reduction vs more iterations vs restarts. To assess the effectiveness of the scheme with progressive reduction of the radius ϵ (red-ϵ) described above, we compare it to the standard scheme (const-ϵ) for a fixed budget. For const-ϵ, to match the budget we use either 300 iterations or 3 random restarts with 100 iterations each, and 300 iterations for red-ϵ. In Fig. 2 we show the robust accuracy achieved by the three attacks with different losses, for ϵ_(∞) ∈ {8/255, 12/255}, on our robust PIR-AT UPerNet + ConvNeXt-T model from Table 1. Our progressive reduction scheme red-ϵ APGD yields the best results (lowest accuracy) for almost every case, with large improvements especially at ϵ_(∞) = 12/255. This shows that this scheme is better suited for generating stronger attacks on semantic segmentation models in comparison to more iterations or restarts used in image classification. The same trend holds for mIoU, see Fig. 4 in Sec. 0.C.1.

### 3.6 Segmentation Ensemble Attack (SEA)

Based on the findings about the complementary properties of the JS and MCE based attacks for different radii ϵ_(∞) targeting average pixel accuracy as well as the MCE-Bal attack targeting mIoU, we use all our three attacks in our Segmentation Ensemble Attack (SEA) for reliable evaluation of the adversarial robustness of semantic segmentation models. SEA consists of one run of 300 iterations with red-ϵ APGD for each of the three losses proposed above, namely ℒ_(MCE), ℒ_(MCE-Bal), and ℒ_(JS), and then taking the worst-case over them (to minimize either accuracy or mIoU).

Worst-case attack in semantic segmentation. While for average pixel accuracy one can simply take for each image the attack which yields the worst image-wise accuracy to get the worst-case accuracy of the ensemble, this is not straightforward for mIoU. As the mIoU is computed using the results on all test images simultaneously, see Eq. (3), the computation of the worst-case mIoU for a given set of attacks is a combinatorial optimization problem as the mIoU cannot be optimized image-wise (the selection of a different attack changes the numerator and denominator). Since solving this combinatorial optimization problem is out of reach, we use a greedy method to find an approximate solution. We select for all images the attack with the smallest mIoU on the test set. Then we iterate the following scheme until there is no improvement in mIoU: in each round we shuffle the images and for each image we check if selecting one of the other attacks would result in a lower mIoU and, if true, update the selected attack for this image. Typically, only 5-10 rounds are sufficient until such a local optimum is found. We highlight that the computational overhead of this scheme is negligible once one has computed true/false positives and false negatives for each image and attack.

Comparison of SEA to prior work. We compare our and SOTA attacks in Tab. 1, at several radii ϵ_(∞) and on various robust and non-robust models on Pascal-Voc and Ade20K, which in turn are based on different architectures: PSPNet, UPerNet and Segmenter. As our novel single attacks already outperform CosPGD and SegPGD, the ensemble SEA of our attacks performs even better: with a single exception regarding mIoU, SEA always achieves the lowest mIoU and accuracy over all six models and different radii. SEA reduces the accuracy compared to CosPGD by up to 28.5% and mIoU by up to 20.9% for the models in Tab. 1. A similar picture holds for SegPGD where SEA always improves in terms of accuracy with maximal gain of 17.2%. For mIoU, SEA improves in all except one case where SegPGD is 0.1% better, whereas the maximal gain is 10.3%. In summary this shows that SEA is a significant improvement in robustness evaluation over previous SOTA attacks and enables a much more reliable robustness evaluation for semantic segmentation. We analyze SEA with several ablation studies in Sec. 0.C.1, e.g. a higher budget of 500 iterations does not yield enough gains to justify the additional computational overhead.

Scope of SEA. The popular AutoAttack [14] for robustness evaluation in image classification has, additionally to its three white-box gradient-based attacks, a black-box attack, in particular to be able to attack defenses which are based on gradient masking. As the creation of an efficient and effective black-box attack for semantic segmentation is a non-trivial research question on its own, we leave this to future work. Due to the lack of a black-box algorithm, SEA can potentially overestimate robustness for defenses exploiting gradient masking as a defense mechanism. Gradient masking is typically not an issue for models obtained with adversarial training. For this class of models, SEA works well as shown by our extensive experiments.

| Setting | Training | Init. | Steps | Backbone | 0 Acc | 0 mIoU | 4/255 Acc | 4/255 mIoU | 8/255 Acc | 8/255 mIoU | 12/255 Acc | 12/255 mIoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pascal-Voc: PSPNet 50 epochs | DDC-AT [48] | clean | 3 | RN-50 | 95.1 | 75.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Pascal-Voc: PSPNet 50 epochs | AT [48] | clean | 3 | RN-50 | 94.0 | 74.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Pascal-Voc: PSPNet 50 epochs | SegPGD-AT [21] | clean | 7 | RN-50 | – | 74.5 | – | – | – | 17.0^(*) | – | – |
| Pascal-Voc: PSPNet 50 epochs | PIR-AT (ours) | robust | 5 | RN-50 | 90.6 | 68.9 | 81.5 | 47.7 | 50.6 | 11.2 | 12.9 | 1.4 |
| Pascal-Voc: UPerNet 50 epochs | AT (ours) | clean | 5 | CN-T | 91.9 | 73.1 | 86.7 | 59.0 | 65.3 | 28.2 | 22.0 | 4.7 |
| Pascal-Voc: UPerNet 50 epochs | PIR-AT (ours) | robust | 5 | CN-T | 92.7 | 75.2 | 88.6 | 64.9 | 71.7 | 34.6 | 28.1 | 5.5 |
| Pascal-Voc: UPerNet 50 epochs | AT (ours) | clean | 5 | CN-S | 92.4 | 74.6 | 88.1 | 61.6 | 68.5 | 30.5 | 23.9 | 4.3 |
| Pascal-Voc: UPerNet 50 epochs | PIR-AT (ours) | robust | 5 | CN-S | 93.1 | 76.6 | 89.1 | 66.0 | 71.0 | 36.4 | 27.6 | 6.2 |
| Ade20K: UPerNet 128 epochs | AT (ours) | clean | 5 | CN-T | 68.0 | 26.1 | 52.2 | 12.8 | 24.5 | 3.3 | 2.5 | 0.2 |
| Ade20K: UPerNet 128 epochs | PIR-AT (ours) | robust | 5 | CN-T | 70.5 | 31.7 | 55.5 | 17.2 | 26.4 | 4.9 | 3.1 | 0.4 |
| Ade20K: Segmenter 128 epochs | AT (ours) | clean | 5 | ViT-S | 67.7 | 26.8 | 49.0 | 11.1 | 25.1 | 3.1 | 4.8 | 0.4 |
| Ade20K: Segmenter 128 epochs | PIR-AT (ours) | robust | 5 | ViT-S | 69.1 | 28.7 | 55.3 | 14.9 | 33.3 | 5.3 | 8.9 | 1.1 |

Table 2: Evaluation of our robust models with SEA on Pascal-Voc and Ade20K. For each model and choice of ϵ_(∞) we report the training details (clean/robust initialization for backbone, number of attack steps and training epochs) and robust average pixel accuracy (white background) and mIoU (grey background) evaluated with SEA. * indicates results reported in [21] evaluated with 100 iterations of SegPGD (models are not available) which is much weaker than SEA.

- Figure 3 panel summary, clean training with UPerNet + ConvNeXt-T backbone: `0 -> Acc 95.9%`, `0.25/255 -> Acc 94.8%`, `0.5/255 -> Acc 75.9%`, `1/255 -> Acc 48.3%`, `2/255 -> Acc 0.0%`.

- Figure 3 panel summary, PIR-AT with UPerNet + ConvNeXt-T backbone: `0 -> Acc 95.5%`, `4/255 -> Acc 94.6%`, `8/255 -> Acc 90.8%`, `12/255 -> Acc 49.2%`, `16/255 -> Acc 0.0%`.

Figure 3: Visualizing adversarial images and their segmentation outputs. We show the perturbed images, corresponding predicted segmentation masks and average accuracy for increasing radii for both the clean and our PIR-AT models. The adversarial images are generated on Pascal-Voc with APGD on ℒ_(Mask-CE). For the clean model even at a smaller radii of 0.5/255, the predicted mask deviates from the ground truth significantly. Whereas for the PIR-AT model the predicted mask is similar to the ground truth even at a high perturbation strength of 8/255. More visualizations can be found in Appendix 0.D.

| Setting | Training | Init. | Steps | Ep. | 0 Acc | 0 mIoU | 4/255 Acc | 4/255 mIoU | 8/255 Acc | 8/255 mIoU | 12/255 Acc | 12/255 mIoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pascal-Voc: UPerNet with ConvNeXt-T backbone | AT | clean | 2 | 50 | 93.4 | 77.4 | 2.6 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| Pascal-Voc: UPerNet with ConvNeXt-T backbone | PIR-AT | robust | 2 | 50 | 92.9 | 75.9 | 86.7 | 60.8 | 50.2 | 21.0 | 9.3 | 2.4 |
| Pascal-Voc: UPerNet with ConvNeXt-T backbone | AT | clean | 2 | 300 | 93.1 | 76.3 | 86.5 | 59.6 | 44.1 | 16.6 | 4.6 | 0.1 |
| Pascal-Voc: UPerNet with ConvNeXt-T backbone | AT | clean | 5 | 50 | 91.9 | 73.1 | 86.7 | 59.0 | 65.3 | 28.2 | 22.0 | 4.7 |
| Pascal-Voc: UPerNet with ConvNeXt-T backbone | PIR-AT | robust | 5 | 50 | 92.7 | 75.2 | 88.6 | 64.9 | 71.7 | 34.6 | 28.1 | 5.5 |
| Pascal-Voc: UPerNet with ConvNeXt-T backbone | AT | clean | 5 | 300 | 92.8 | 75.5 | 88.6 | 64.8 | 71.5 | 35.1 | 23.7 | 5.1 |
| Ade20K: UPerNet with ConvNeXt-T backbone | AT | clean | 2 | 128 | 73.4 | 36.4 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Ade20K: UPerNet with ConvNeXt-T backbone | PIR-AT | robust | 2 | 128 | 72.0 | 34.7 | 45.9 | 15.1 | 5.9 | 1.9 | 0.0 | 0.0 |
| Ade20K: UPerNet with ConvNeXt-T backbone | PIR-AT | robust | 5 | 32 | 68.8 | 25.2 | 56.2 | 13.7 | 30.6 | 5.0 | 4.5 | 0.6 |
| Ade20K: UPerNet with ConvNeXt-T backbone | AT | clean | 5 | 128 | 68.0 | 26.1 | 52.2 | 12.8 | 24.6 | 3.3 | 2.5 | 0.2 |
| Ade20K: UPerNet with ConvNeXt-T backbone | PIR-AT | robust | 5 | 128 | 70.5 | 31.7 | 55.5 | 17.2 | 26.4 | 4.9 | 3.1 | 0.4 |
| Ade20K: Segmenter with ViT-S backbone | PIR-AT | robust | 5 | 32 | 68.1 | 26.0 | 55.5 | 14.2 | 34.2 | 5.3 | 8.7 | 0.9 |
| Ade20K: Segmenter with ViT-S backbone | AT | clean | 5 | 128 | 67.7 | 26.8 | 49.0 | 11.1 | 25.1 | 3.1 | 4.8 | 0.4 |
| Ade20K: Segmenter with ViT-S backbone | PIR-AT | robust | 5 | 128 | 69.1 | 28.7 | 55.3 | 14.9 | 33.3 | 5.3 | 8.9 | 1.1 |

Table 3: Ablation study AT vs PIR-AT. We show the effect of varying the number of attack steps and training epochs on the robustness (measured with Acc and mIoU at various radii) of the models trained with AT and PIR-AT. Our PIR-AT achieves similar or better robustness than AT at significantly reduced computational cost for all datasets and architectures.

## 4 Adversarially Robust Segmentation Models

Adversarial training (AT) [30] and its variants are the established techniques to get adversarially robust image classifiers. One major drawback is the significantly longer training time due to the adversarial attack steps (k steps imply a factor of k + 1 higher cost). This can be prohibitive for large backbones and decoders in semantic segmentation models. As a remedy we propose Pre-trained ImageNet Robust Models AT (PIR-AT), which can reduce the training time by a factor of upto 6 while improving the SOTA of robust semantic segmentation models.

Experimental setup. We use PGD for adversarial training with the cross-entropy loss and an ℓ_(∞)-threat model of radius ϵ_(∞) = 4/255 (as used by robust classifiers on ImageNet). We tried AT with the losses from Sec. 3.4 but got no improvements over the cross-entropy loss. This is to be expected as the good properties of the JS-divergence for attacking a semantic segmentation model are also detrimental for training such a model. For training configuration, we mostly follow standard practices for each architecture [49, 29, 40], in particular for the number of epochs (see Appendix 0.B for training and evaluation details). All robustness evaluations are done with SEA on the entire validation set.

### 4.1 PIR-AT: robust models via robust initialization

When training clean and robust semantic segmentation models it is common practice to initialize the backbone with a clean classifier pre-trained on ImageNet [40, 29], which are not robust to adversaries even at small radii (e.g. ϵ_(∞) = 1/255), see the top row in Fig. 3 for an illustration. In contrast, we propose for training robust models to use ℓ_(∞)-robust ImageNet classifiers (see Sec. 0.B.3 for specific models) as initialization of the backbone, whereas the decoder is always initialized randomly: we denote this approach as Pre-trained ImageNet Robust Models AT (PIR-AT). The resulting PIR-AT models are robust even at a large radius such as 8/255, see bottom row in Fig. 3. This seemingly small change has huge impact on the robustness of the model. We show in Tab. 2 a direct comparison of AT and PIR-AT using the same number of adversarial steps and training epochs across different decoders and architectures for the backbones. In all cases PIR-AT outperforms AT in clean and robust accuracy. Up to our knowledge these are also the first robust models for Ade20K. Regarding Pascal-Voc, we note that DDC-AT from [48] as well as their AT-model are completely non-robust. SegPGD-AT, trained for ϵ_(∞) = 8/255 and with the larger attack budget of 7 steps, is seemingly more robust than our PSPNet, but it is evaluated with only 100 iterations of SegPGD, which is significantly weaker than SEA and therefore its robustness could be overestimated (this model is not publicly available). However, our UPerNet + ConvNeXt-T (CN-T) outperforms SegPGD-AT by at least 17.6% in mIoU at ϵ_(∞) = 8/255 even though it is trained for 4/255. Interestingly, the large gains in robustness do not degrade much the clean performance, which is a typical drawback of adversarial training. Our results for ConvNeXt-S (CN-S) show that this also scales to larger backbones.

### 4.2 Ablation study of AT vs PIR-AT

In Tab. 3 we provide a detailed comparison of AT vs PIR-AT for different number of adversarial steps and training epochs. On Pascal-Voc, for 2 attack steps, AT with clean initialization for 50 epochs does not lead to any robustness. This is different to image classification where 50 epochs 2-step AT are sufficient to get robust classifiers on ImageNet [39]. In contrast 2-step PIR-AT yields a robust model and even outperforms 300 epochs of AT with clean initialization in terms of robustness at all ϵ_(∞) while being 6 times faster to train. This shows the significance of the initialization. For 5 attack steps we see small improvements of robustness at 4/255 compared to 2 steps but much larger ones at 8/255. Again, 50 epochs of PIR-AT mostly outperform the 300 epochs of AT with clean initialization. Our findings generalize to Ade20K and across architectures (UPerNet + ConvNeXt-T and Segmenter+ViT-S): 32 epochs of PIR-AT outperform 128 epochs of AT with clean initialization in terms of robustness (4 times faster). Robust ImageNet classifiers are now available for different architectures and sizes, thus PIR-AT should become standard to train robust semantic segmentation models.

## 5 Conclusion

We have proposed novel attacks for semantic segmentation which take into account the specific properties of this problem. We obtain significant improvements over SOTA attacks and SEA, an ensemble of our attacks, yields by large margin the best results regarding accuracy and mIoU. Moreover, we show how to train segmentation models with SOTA robustness, even at a significantly reduced cost by using adversarially pre-trained ImageNet classifiers. We hope that the availability of our robust models, together with code for training and attacks, will foster research in robust semantic segmentation.

Limitations. We consider SEA an important step towards a strong robustness evaluation of semantic segmentation models. However, similar to AutoAttack [14], white-box attacks should be complemented by strong black-box attacks which we leave to future work. Moreover, several techniques, e.g. using different loss functions, unlabeled and synthetic data, adversarial weight perturbations, etc., have been shown effective to achieve more robust classifiers, and might be beneficial for segmentation too, but this is out of scope for this work.

Acknowledgements

We thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting NDS. We acknowledge support from the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy (EXC number 2064/1, project number 390727645), as well as in the priority program SPP 2298, project number 464101476. We are also thankful for the support of Open Philanthropy and the Center for AI Safety Compute Cluster. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the sponsors.

References

-   [1] Agnihotri, S., Keuper, M.: Cospgd: a unified white-box adversarial attack for pixel-wise prediction tasks. arXiv preprint arXiv:2302.02213 (2023)
-   [2] Arnab, A., Miksik, O., Torr, P.H.: On the robustness of semantic segmentation models to adversarial attacks. In: CVPR (2018)
-   [3] Athalye, A., Carlini, N., Wagner, D.: Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In: ICML (2018)
-   [4] Bai, Y., Mei, J., Yuille, A., Xie, C.: Are transformers more robust than CNNs? In: NeurIPS (2021)
-   [5] Bao, H., Dong, L., Piao, S., Wei, F.: Beit: Bert pre-training of image transformers. In: ICLR (2022)
-   [6] Biggio, B., Corona, I., Maiorca, D., Nelson, B., Šrndić, N., Laskov, P., Giacinto, G., Roli, F.: Evasion attacks against machine learning at test time. In: ECML/PKKD (2013)
-   [7] Brown, T.B., Mané, D., Roy, A., Abadi, M., Gilmer, J.: Adversarial patch. In: NeurIPS 2017 Workshop on Machine Learning and Computer Security (2017)
-   [8] Carlini, N., Wagner, D.: Towards evaluating the robustness of neural networks. In: IEEE Symposium on Security and Privacy (2017)
-   [9] Chen, P.Y., Sharma, Y., Zhang, H., Yi, J., Hsieh, C.J.: Ead: Elastic-net attacks to deep neural networks via adversarial examples. In: AAAI (2018)
-   [10] Cho, S., Jun, T.J., Oh, B., Kim, D.: Dapas: Denoising autoencoder to prevent adversarial attack in semantic segmentation. In: IJCNN. IEEE (2020)
-   [11] Cisse, M., Adi, Y., Neverova, N., Keshet, J.: Houdini: Fooling deep structured prediction models. In: NeurIPS (2017)
-   [12] Croce, F., Andriushchenko, M., Sehwag, V., Debenedetti, E., Flammarion, N., Chiang, M., Mittal, P., Hein, M.: Robustbench: a standardized adversarial robustness benchmark. In: NeurIPS Datasets and Benchmarks Track (2021)
-   [13] Croce, F., Andriushchenko, M., Singh, N.D., Flammarion, N., Hein, M.: Sparse-rs: a versatile framework for query-efficient sparse black-box adversarial attacks. In: AAAI (2022)
-   [14] Croce, F., Hein, M.: Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In: ICML (2020)
-   [15] Croce, F., Hein, M.: Mind the box: l₁-apgd for sparse adversarial attacks on image classifiers. In: ICML (2021)
-   [16] Debenedetti, E., Sehwag, V., Mittal, P.: A light recipe to train robust vision transformers. In: IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). pp. 225–253 (2023)
-   [17] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021)
-   [18] Everingham, M., Van Gool, L., Williams, C.K., Winn, J., Zisserman, A.: The pascal visual object classes (voc) challenge. International journal of computer vision 88, 303–338 (2010)
-   [19] Goodfellow, I.J., Shlens, J., Szegedy, C.: Explaining and harnessing adversarial examples. In: ICLR (2015)
-   [20] Grosse, K., Papernot, N., Manoharan, P., Backes, M., McDaniel, P.: Adversarial perturbations against deep neural networks for malware classification. arXiv preprint arXiv:1606.04435 (2016)
-   [21] Gu, J., Zhao, H., Tresp, V., Torr, P.H.: Segpgd: An effective and efficient adversarial attack for evaluating and boosting segmentation robustness. In: ECCV (2022)
-   [22] Hariharan, B., Arbeláez, P., Bourdev, L., Maji, S., Malik, J.: Semantic contours from inverse detectors. In: ICCV (2011)
-   [23] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016)
-   [24] Jin, D., Jin, Z., Zhou, J.T., Szolovits, P.: Is BERT really robust? natural language attack on text classification and entailment. In: AAAI (2019)
-   [25] Kang, X., Song, B., Du, X., Guizani, M.: Adversarial attacks for image segmentation on multiple lightweight models. IEEE Access 8, 31359–31370 (2020)
-   [26] Kapoor, N., Bär, A., Varghese, S., Schneider, J.D., Hüger, F., Schlicht, P., Fingscheidt, T.: From a fourier-domain perspective on adversarial examples to a wiener filter defense for semantic segmentation. In: IJCNN (2021)
-   [27] Laidlaw, C., Singla, S., Feizi, S.: Perceptual adversarial robustness: Defense against unseen threat models. In: ICLR (2021)
-   [28] Liu, C., Dong, Y., Xiang, W., Yang, X., Su, H., Zhu, J., Chen, Y., He, Y., Xue, H., Zheng, S.: A comprehensive study on robustness of image classification models: Benchmarking and rethinking. arXiv preprint, arXiv:2302.14301 (2023)
-   [29] Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. CVPR (2022)
-   [30] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A.: Towards deep learning models resistant to adversarial attacks. In: ICLR (2018)
-   [31] Metzen, J.H., Chaithanya Kumar, M., Brox, T., Fischer, V.: Universal adversarial perturbations against semantic image segmentation. In: ICCV (2017)
-   [32] Mopuri, K.R., Ganeshan, A., Babu, R.V.: Generalizable data-free objective for crafting universal adversarial perturbations. IEEE transactions on pattern analysis and machine intelligence 41(10), 2452–2465 (2018)
-   [33] Nesti, F., Rossolini, G., Nair, S., Biondi, A., Buttazzo, G.: Evaluating the robustness of semantic segmentation for autonomous driving against real-world adversarial patch attacks. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 2280–2289 (2022)
-   [34] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S.: Pytorch: An imperative style, high-performance deep learning library. In: NeurIPS (2019)
-   [35] Rony, J., Hafemann, L.G., Oliveira, L.S., Ayed, I.B., Sabourin, R., Granger, E.: Decoupling direction and norm for efficient gradient-based l2 adversarial attacks and defenses. In: CVPR (2019)
-   [36] Rony, J., Pesquet, J.C., Ben Ayed, I.: Proximal splitting adversarial attacks for semantic segmentation. In: CVPR (2023)
-   [37] Salman, H., Ilyas, A., Engstrom, L., Kapoor, A., Madry, A.: Do adversarially robust imagenet models transfer better? In: NeurIPS (2020)
-   [38] Shen, G., Mao, C., Yang, J., Ray, B.: Advspade: Realistic unrestricted attacks for semantic segmentation. arXiv preprint arXiv:1910.02354 (2019)
-   [39] Singh, N.D., Croce, F., Hein, M.: Revisiting adversarial training for imagenet: Architectures, training and generalization across threat models. In: NeurIPS (2023)
-   [40] Strudel, R., Garcia, R., Laptev, I., Schmid, C.: Segmenter: Transformer for semantic segmentation. In: CVPR (2021)
-   [41] Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., Fergus, R.: Intriguing properties of neural networks. In: ICLR (2014)
-   [42] Tramèr, F., Carlini, N., Brendel, W., Madry, A.: On adaptive attacks to adversarial example defenses. In: NeurIPS (2020)
-   [43] Wightman, R.: Pytorch image models. https://github.com/rwightman/pytorch-image-models (2019). https://doi.org/10.5281/zenodo.4414861
-   [44] Wong, E., Schmidt, F.R., Kolter, J.Z.: Wasserstein adversarial examples via projected sinkhorn iterations. In: ICML (2019)
-   [45] Xiao, C., Deng, R., Li, B., Yu, F., Liu, M., Song, D.: Characterizing adversarial examples based on spatial consistency information for semantic segmentation. In: ECCV (2018)
-   [46] Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. In: ECCV (2018)
-   [47] Xie, C., Wang, J., Zhang, Z., Zhou, Y., Xie, L., Yuille, A.: Adversarial examples for semantic segmentation and object detection. In: ICCV (2017)
-   [48] Xu, X., Zhao, H., Jia, J.: Dynamic divide-and-conquer adversarial training for robust semantic segmentation. In: ICCV (2021)
-   [49] Zhao, H., Shi, J., Qi, X., Wang, X., Jia, J.: Pyramid scene parsing network. In: CVPR (2017)
-   [50] Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., Torralba, A.: Semantic understanding of scenes through the ade20k dataset. IJCV (2019)

Contents of the Appendix

1.  1.
    Broader Impact
2.  2.
## Appendix 0.A …Omitted proofs
3.  3.
## Appendix 0.B …Experimental and evaluation details
4.  4.
## Appendix 0.C …Additional SEA experiments, ablations and comparisons
5.  5.
## Appendix 0.D …Visualizations of adversarial images generated by SEA

Broader Impact

We propose new techniques to test the robustness of segmentation models to adversarial attacks. While we consider it important to estimate the vulnerability of existing systems, such methods might potentially be used by malicious actors. However, we also provide insights on how to obtain, at limited computational cost, models which are robust to such perturbations.

## Appendix 0.A Proof of the Properties of Cross-Entropy, and the Jensen-Shannon-Divergence loss

Cross-entropy loss:
The cross-entropy is given as: ℒ_(CE)(p,e_(y)) = −log p_(y), and has gradient

  -- ------------------------------------------------------------------------------------------------------------------------------------- --
     $${\frac{\partial{\mathcal{L}_{\text{CE}}{({\mathbf{u}},e_{y})}}}{\partial u_{t}} = {{- \delta_{yt}} + {{\mathbf{p}}_{t}{(u)}}}}.$$
  -- ------------------------------------------------------------------------------------------------------------------------------------- --

We note that

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${\left\| {{\nabla_{u}\mathcal{L}_{\text{CE}}}{({\mathbf{u}},e_{y})}} \right\|_{2}^{2} = {{\sum\limits_{t \neq y}{\mathbf{p}}_{t}^{2}} + {({1 - {\mathbf{p}}_{y}})}^{2}}}.$$
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

As 0 ≤ p_(t) ≤ 1, it holds

  -- ---------------------------------------------------------------------------------------------------------------------------- --
     $${{\sum\limits_{t \neq y}{\mathbf{p}}_{t}^{2}} \leq {\sum\limits_{t \neq y}{\mathbf{p}}_{t}} = {1 - {\mathbf{p}}_{y}}}.$$
  -- ---------------------------------------------------------------------------------------------------------------------------- --

Moreover, the point of minimal ℓ₂-distance on the surface of the ℓ₁-ball with radius 1 − p_(y) has equal components, and thus

  -- ------------------------------------------------------------------------------------------------------- --
     $${{\sum\limits_{t \neq y}{\mathbf{p}}_{t}^{2}} \geq \frac{{({1 - {\mathbf{p}}_{y}})}^{2}}{K - 1}},$$
  -- ------------------------------------------------------------------------------------------------------- --

which yields

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{\frac{K}{K - 1}{({1 - {\mathbf{p}}_{y}})}^{2}} \leq \left\| {{\nabla_{u}\mathcal{L}_{\text{CE}}}{({\mathbf{u}},e_{y})}} \right\|_{2}^{2} \leq {{1 - {\mathbf{p}}_{y}} + {({1 - {\mathbf{p}}_{y}})}^{2}}}.$$
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

We note that both lower and upper bounds are monotonically increasing as p_(y) → 0.

Jensen-Shannon divergence:
The Jensen-Shannon-divergence between the predicted distribution p and the label distribution q is given by

  -- --------------------------------------------------------------- --
     D_(JS)(p∥q) = (D_(KL)(p∥m)+D_(KL)(q∥m))/2, with  m = (p+q)/2,
  -- --------------------------------------------------------------- --

Assuming that we have a one-hot label encoding q = e_(y) (where e_(y) is the y-th cartesian coordinate vector), one gets

  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${{D_{\text{JS}}{({{\mathbf{p}} \parallel e_{y}})}} = {{\frac{1}{2}{\log\left( \frac{2}{1 + {\mathbf{p}}_{y}} \right)}} + {\frac{1}{2}{\sum\limits_{i = 1}^{K}{{\mathbf{p}}_{i}{\log\left( \frac{2{\mathbf{p}}_{i}}{\delta_{yi} + {\mathbf{p}}_{i}} \right)}}}}}}.$$
  -- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

Then

  -- ---------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $\frac{\partial{D_{\text{JS}}{({{\mathbf{p}} \parallel e_{y}})}}}{\partial{\mathbf{p}}_{r}}$   $= {\frac{1}{2}\left\lbrack {{{- {\frac{1}{1 + {\mathbf{p}}_{y}}\delta_{yr}}} + {\log\left( \frac{2{\mathbf{p}}_{r}}{\delta_{yr} + {\mathbf{p}}_{r}} \right)} + 1} - \frac{{\mathbf{p}}_{r}}{\delta_{yr} + {\mathbf{p}}_{r}}} \right\rbrack}$
                                                                                                    $= {\frac{1}{2}\begin{cases}
                                                                                                    {\log\left( \frac{2{\mathbf{p}}_{y}}{1 + {\mathbf{p}}_{y}} \right)} & {{{\text{~if~}r} = y},} \\
                                                                                                    {\log{(2)}} & {\text{~else}.}
                                                                                                    \end{cases}}$
  -- ---------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

Given the logits u we use the softmax function

  -- -------------------------------------------------------------------------------------------- --
     $${{{\mathbf{p}}_{r} = \frac{e^{u_{r}}}{\sum_{t = 1}^{K}e^{u_{t}}}},{r = {1,\ldots,K}}},$$
  -- -------------------------------------------------------------------------------------------- --

to obtain the predicted probability distribution p. One can compute:

  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $${\frac{\partial{\mathbf{p}}_{r}}{\partial u_{t}} = {{{\delta_{rt}{\mathbf{p}}_{t}} - {{\mathbf{p}}_{r}{\mathbf{p}}_{t}}}\quad\Longrightarrow}}\quad{{\sum\limits_{r = 1}^{K}\frac{\partial{\mathbf{p}}_{r}}{\partial u_{t}}} = 0}$$
  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

Then

  -- ------------------------------------------------------------------------------------ ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
     $\frac{\partial{D_{\text{JS}}{({{\mathbf{p}} \parallel e_{y}})}}}{\partial u_{t}}$   $= {\sum\limits_{r = 1}^{K}{\frac{\partial{D_{\text{JS}}{({{\mathbf{p}} \parallel e_{y}})}}}{\partial{\mathbf{p}}_{r}}\frac{\partial{\mathbf{p}}_{r}}{\partial u_{t}}}} = {\frac{1}{2}\left\lbrack {{{\log\left( \frac{2{\mathbf{p}}_{y}}{1 + {\mathbf{p}}_{y}} \right)}\frac{\partial{\mathbf{p}}_{y}}{\partial u_{t}}} + {{\log{(2)}}{\sum\limits_{r \neq y}\frac{\partial{\mathbf{p}}_{r}}{\partial u_{t}}}}} \right\rbrack}$
                                                                                          $= {\frac{1}{2}\left( {{\log\left( {\frac{2{\mathbf{p}}_{y}}{1 + {\mathbf{p}}_{y}}\frac{\partial{\mathbf{p}}_{y}}{\partial u_{t}}} \right)} - {{\log{(2)}}\frac{\partial{\mathbf{p}}_{y}}{\partial u_{t}}}} \right)}$
                                                                                          $= {\frac{1}{2}\left( {{\log\left( \frac{{\mathbf{p}}_{y}}{1 + {\mathbf{p}}_{y}} \right)}\left\lbrack {{\delta_{yt}{\mathbf{p}}_{y}} - {{\mathbf{p}}_{y}{\mathbf{p}}_{t}}} \right\rbrack} \right)}$
                                                                                          $= {\frac{1}{2}\left( {{\mathbf{p}}_{y}{\log\left( \frac{{\mathbf{p}}_{y}}{1 + {\mathbf{p}}_{y}} \right)}\left\lbrack {\delta_{yt} - {\mathbf{p}}_{t}} \right\rbrack} \right)}$
  -- ------------------------------------------------------------------------------------ ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

Noting that lim_(x → 0)xlog (x) = 0 we get the result that: ${{\lim\limits_{{\mathbf{p}}_{y}\rightarrow 0}\frac{\partial{D_{\text{JS}}{({{\mathbf{p}} \parallel e_{y}})}}}{\partial u_{t}}} = 0}.$

Thus the ℒ_(JS) loss automatically down-weights contributions from mis-classified pixels and thus pixels which are still correctly classified get a higher weight in the gradient.

Discussion. The (theoretical) discussion of benefits and weaknesses for each loss in Sec. 3.4 suggests that one main difference among losses is how they balance the weight of different pixels in the objective function. On one extreme, the plain cross-entropy maximizes the loss for all pixels independently of whether they are misclassified, and assigns them the same importance. Conversely, the masked losses exclude (via the mask) the misclassified pixels from the objective function, with the danger of reverting back the successful perturbations. As middle ground, losses like the JS divergence assign a weight to each pixel based on how “confidently” they are misclassified. We conjecture that for radii where robustness is low, masked losses help focusing on the remaining pixels, and already misclassified pixels are hardly reverted since they are far from the decision boundary. Conversely, at smaller radii achieving confident misclassification is harder (since the perturbations are smaller), and most pixels are still correctly classified or misclassified but close to the decision boundary: then it becomes more important to balance all of them in the loss, hence losses like JS divergence are more effective. This hypothesis is in line with the empirical results in Tab. 1 and Tab. 6.

## Appendix 0.B Experimental Details

We here provide additional details about both attacks and training scheme used in the experiments in the main part.

0.B.1 Attacks for semantic segmentation

Baselines. Since [21, 1] do not provide code for their methods, we re-implement both SegPGD and CosPGD following the indications in the respective papers and personal communication with the authors of CosPGD. In the comparison in Table 1, we use PGD with step size (8e-4, 9e-4, 1e-3, 2e-3, 3e-3, 5e-3, 6e-3) for radii (0.25/255, 0.5/255, 1/255, 2/255, 4/255, 8/255, 12/255) resp. for both CosPGD and SegPGD for 300 iterations each. The step size selection was done via a small grid-search in [2e-3, 3e-3, 5e-3, 6e-3, 1e-4] for ϵ = 4/255 and 8/255, the values for other radii were extrapolated from these. Moreover, at the end we select for each image the iterate with highest loss (strongest yet generated adversary).

APGD with masked losses. Since APGD relies on the progression of the objective function value to e.g. select the step size, using losses which mask the mis-classified pixels might be problematic, since the loss is not necessarily monotonic. Then, in practice we only apply the mask when computing the gradient at each iteration.

0.B.2 Training robust models

In the following, we detail the employed network architectures, as well as our training procedure for the utilized datasets. All experiments are conducted in multi-GPU setting with PyTorch [34] library. For adversarial training we use PGD at ϵ = 4/255 and step size 0.01. While training clean or adversarially, the backbones are initialized with publicly available ImageNet pre-trained models, source of which are listed in Table 5.

| Group | Configuration | Pascal-Voc PSPNet | Pascal-Voc UPerNet | Ade20K UPerNet | Ade20K Segmenter |
| --- | --- | --- | --- | --- | --- |
| DATA | Base size | 512 | 512 | 520 | 520 |
| DATA | Crop size | 473x473 | 473x473 | 512x512 | 512x512 |
| DATA | Random Horizontal Flip | ✓ | ✓ | ✓ | ✓ |
| DATA | Random Gaussian Blur | ✓ | ✓ | ✓ | ✓ |
| TRAINING | Optimizer | SGD | AdamW | AdamW | SGD |
| TRAINING | Base learning rate | 5e-4 | 1e-3 | 1e-3 | 2e-3 |
| TRAINING | Weight decay | 0.0 | 1e-2 | 1e-2 | 1e-2 |
| TRAINING | Batch size | 16x8 | 16x8 | 16x8 | 16x8 |
| TRAINING | Epochs | 50/300 | 50/300 | 32/128 | 32/128 |
| TRAINING | Warmup epochs | 5/30 | 5/30 | 5/20 | 5/20 |
| TRAINING | Momentum | 0.9 | 0.9, 0.999 | 0.9, 0.999 | 0.9 |
| TRAINING | LR schedule | poly dec. | poly dec. | poly dec. | poly dec. |
| TRAINING | Warmup schedule | linear | linear | linear | linear |
| TRAINING | Schedule power | 0.9 | 1.0 | 1.0 | 0.9 |
| TRAINING | LR ratio (Enc:Dec) | 1:10 | ✗ | ✗ | ✗ |
| TRAINING | Auxiliary loss weight | 0.4 | 0.4 | 0.4 | – |

Table 4: Training and data configurations. For all the models trained in this work, we list according to the dataset, the training and dataset configurations. Warmup epochs are scaled depending on the total number of epochs. Poly dec. is the polynomially decaying schedule, from [49]. The setup stays the same across all setups of adversarial training (clean init./robust init. or 2 vs 5 step).

| Architecture | Backbone | Robust | Source | ImageNet clean acc. | ImageNet robust acc. at ℓ_(∞)=4/255 |
| --- | --- | --- | --- | --- | --- |
| UPerNet | ConvNeXt-T + ConvStem | ✗ | [39] | 80.9% | 0.0% |
| UPerNet | ConvNeXt-T + ConvStem | ✓ | [39] | 72.7% | 49.5% |
| UPerNet | ConvNeXt-S + ConvStem | ✓ | [39] | 74.1% | 52.4% |
| Segmenter | ViT-S | ✗ | [43] | 81.2% | 0.0% |
| Segmenter | ViT-S | ✓ | [39] | 69.2% | 44.4% |
| PSPNet | ResNet-50 | ✓ | [37] | 64.0% | 35.0% |

Table 5: Source of our pre-trained backbones. We employ the same backbone for both Pascal-Voc and Ade20K. The robust column indicates if the backbone used is adversarially robust for ImageNet and we also list the ImageNet clean and robust accuracy at ℓ_(∞)-radius of 4/255.

Model architectures. Semantic segmentation model architectures have adapted to use image classifiers in their backbone. UPerNet coupled with ConvNeXt [29] and transformer models like ViT [17] with Segmenter [40] achieve SOTA segmentation results. We choose UPerNet and Segmenter architectures for our experiments with ConvNeXt and ViT as their respective backbones. For direct comparison to existing robust segmentation works [21, 48] which only train with a PSPNet [49], we also train a PSPNet with a ResNet-50 backbone (see Tables 2 and 6).  Tab. 4 reports the training and data related information about the various architectures and the backbones used.

UPerNet with ConvNeXt backbone. For both clean and robust initialization setups, we use the publically available ImageNet-1k pre-trained weights¹¹1https://github.com/nmndeep/revisiting-at from [39], which achieve SOTA robustness for ℓ_(∞)-threat model at ϵ = 4/255. They propose some architectural changes, notably replacing PatchStem with a ConvStem in their most robust ConvNeXt models, and we keep these changes intact in our UPerNet models, we always use a ConvNeXt with ConvStem in this work. We highlight that ConvNeXt-T, when adversarially trained for classification on ImageNet, attains significantly higher robustness than ResNet-50 at a similar parameter and FLOPs count. For example, at ϵ_(∞) = 4/255, the ConvNeXt-T we use has 49.5% of robust accuracy, while ResNet-50 is reported to achieve around 35% [37, 4]. This supports choosing ConvNeXt as backbone for obtaining robust segmentation models with the UPerNet architecture. For UPerNet with the ConvNeXt backbone, we use the training setup from [29], listed in Tab. 4. We also use the same values of 0.4 or 0.3 for stochastic depth coefficient depending on the backbone, same as the original work.²²2https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/configs/convnext We do not use heavier augmentations and Layer-Decay [5] optimizer as done by [29].

Segmenter with ViT backbone. Testing with Segmenter also enables a further comparison across model size as Segmenter with a ViT-S backbone is less than half the size (26 million parameters) of UPerNet with a ConvNeXt-T backbone (60 million parameters). We define the training setup in Table 4, which is similar to the setup used by [40]. The decoder is a Mask transformer and is randomly initialized. Note that [40] predominantly use ImageNet pre-trained classifiers at resolution of 384x384, whereas we use 224x224 resolution as no robust models at the higher resolution are available.

PSPNet with ResNet backbone. As prior works [48, 21] use a PSPNet with a ResNet [23] backbone to test their robustness evaluations, we also train the same model for the Pascal-Voc dataset. Both DDCAT [48] and SegPGD-AT [21] use a split of 50% clean and 50% adversarial inputs for training. Instead for PIR-AT with PSPNet, we just use adversarial inputs. Due to this change, and due to the fact that we initialize PIR-AT with ImageNet pre-trained ResNet-50 (RN50), we slightly deviate from the standard training parameters (learning rate, weight decay, warmup epochs) as in the original PSPNet work [49]. The detailed training setup is listed in Tab. 4.

Training setup for Pascal-Voc. We use the augmentation setup from [22]. Our training set comprises of 8498 images and we validate on the original Pascal-Voc validation set of 1449 images. Data and training configurations are detailed in Tab. 4. Adversarial training is done with either 2 or 5 steps of PGD with the cross-entropy loss. Unlike some other works in literature, we train for 21 classes (including the background class).

[Refer to caption]

Figure 4: Comparison of const-ϵ- and red-ϵ optimization schemes for mIoU. Balanced attack accuracy for the robust PIR-AT trained UPerNet + ConvNeXt-T model from Tab. 2 trained on Pascal-Voc, across different losses for the same iteration budget. The radius reduction (red-ϵ) scheme performs best across all losses, and ϵ_(∞) and even the worst-case over all losses improves.

| Model | ϵ_(∞) | ℒ_(MCE) Acc | ℒ_(MCE) mIoU | ℒ_(MCE-Bal) Acc | ℒ_(MCE-Bal) mIoU | ℒ_(JS) Acc | ℒ_(JS) mIoU | SEA Acc | SEA mIoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PSPNet ResNet50, PIR-AT, 50 epochs, Pascal-Voc | 4/255 | 83.3 | 48.6 | 84.7 | 49.9 | 81.8 | 47.8 | 81.5 | 47.7 |
| PSPNet ResNet50, PIR-AT, 50 epochs, Pascal-Voc | 8/255 | 53.4 | 13.5 | 56.4 | 12.2 | 53.7 | 14.1 | 50.6 | 11.2 |
| PSPNet ResNet50, PIR-AT, 50 epochs, Pascal-Voc | 12/255 | 14.9 | 2.3 | 17.6 | 1.7 | 20.7 | 4.1 | 12.9 | 1.4 |
| UPerNet ConvNeXt-T, PIR-AT, 50 epochs, Pascal-Voc | 4/255 | 89.2 | 65.9 | 90.4 | 67.4 | 88.7 | 64.9 | 88.6 | 64.9 |
| UPerNet ConvNeXt-T, PIR-AT, 50 epochs, Pascal-Voc | 8/255 | 74.0 | 40.6 | 77.5 | 38.4 | 73.9 | 41.3 | 71.7 | 34.6 |
| UPerNet ConvNeXt-T, PIR-AT, 50 epochs, Pascal-Voc | 12/255 | 31.5 | 10.3 | 36.9 | 6.7 | 38.6 | 15.1 | 28.1 | 5.5 |
| UPerNet ConvNeXt-S, PIR-AT, 50 epochs, Pascal-Voc | 4/255 | 89.7 | 67.5 | 90.9 | 68.9 | 89.3 | 66.7 | 89.1 | 66.0 |
| UPerNet ConvNeXt-S, PIR-AT, 50 epochs, Pascal-Voc | 8/255 | 73.6 | 41.0 | 77.5 | 36.9 | 74.3 | 42.7 | 71.0 | 36.4 |
| UPerNet ConvNeXt-S, PIR-AT, 50 epochs, Pascal-Voc | 12/255 | 31.2 | 10.7 | 36.9 | 7.5 | 39.0 | 15.6 | 27.6 | 6.2 |
| UPerNet ConvNeXt-T, PIR-AT, 128 epochs, Ade20K | 4/255 | 56.8 | 20.0 | 58.2 | 17.9 | 55.9 | 18.9 | 55.5 | 17.2 |
| UPerNet ConvNeXt-T, PIR-AT, 128 epochs, Ade20K | 8/255 | 28.5 | 6.6 | 31.1 | 5.3 | 28.5 | 7.2 | 26.4 | 4.9 |
| UPerNet ConvNeXt-T, PIR-AT, 128 epochs, Ade20K | 12/255 | 3.7 | 0.9 | 4.5 | 0.9 | 5.2 | 1.1 | 3.1 | 0.4 |
| UPerNet ConvNeXt-S, PIR-AT, 128 epochs, Ade20K | 4/255 | 58.6 | 20.4 | 59.8 | 18.6 | 57.6 | 19.4 | 56.8 | 17.9 |
| UPerNet ConvNeXt-S, PIR-AT, 128 epochs, Ade20K | 8/255 | 31.3 | 8.1 | 33.3 | 5.8 | 30.9 | 7.7 | 28.7 | 5.4 |
| UPerNet ConvNeXt-S, PIR-AT, 128 epochs, Ade20K | 12/255 | 4.6 | 1.1 | 5.4 | 0.8 | 6.2 | 1.3 | 3.1 | 0.6 |
| Segmenter ViT-S, PIR-AT, 128 epochs, Ade20K | 4/255 | 56.9 | 17.8 | 57.6 | 15.6 | 55.6 | 16.6 | 55.3 | 14.9 |
| Segmenter ViT-S, PIR-AT, 128 epochs, Ade20K | 8/255 | 36.2 | 8.5 | 37.8 | 5.6 | 34.2 | 7.7 | 33.3 | 5.4 |
| Segmenter ViT-S, PIR-AT, 128 epochs, Ade20K | 12/255 | 10.5 | 2.2 | 11.7 | 1.3 | 11.2 | 2.2 | 8.9 | 1.1 |

Table 6: Component analysis for SEA. We show the individual performance (Acc) of the runs of APGD (red- ϵ) with each loss in SEA for both Pascal-Voc and Ade20K on 5-step robust models. The best results, among either individual runs, are in bold.

Training setup for Ade20K. We use the full standard training and validation sets from [50]. Adversarial training is done with either 2 or 5 steps of PGD with the cross-entropy loss. Unlike the original work we train with 151 classes (including the background class).

0.B.3 Initialization with pre-trained backbones

PIR-AT uses pre-trained ImageNet models as an initialization for the backbone. Note that in the semantic segmentation literature most modern works [29, 40] use clean ImageNet pre-trained models as initialization for the backbone, making ours a natural choice. The robust models are sourced from [39] (see Tab. 5), and more are available e.g. in RobustBench [12], thus they do not cost us any additional pre-training. One can further reduce the cost of pre-training by using robust models trained for either 1-step [16] or 2-step [39] adversarial training, which is the common budget for robust ImageNet training. For our UPerNet + ConvNeXt-S PIR-AT model for Pascal-Voc, we use the 2-step 50 epoch ImageNet trained model from [39] as initialization. Using such low-cost pre-trained backbones works well, as this model in Tab. 2 achieves better or similar robust accuracy as the 300 epoch 2-step ImageNet pre-trained ConvNeXt-T in the same table.

## Appendix 0.C Additional Experiments and Discussion

We present additional studies of the properties of SEA and of the robust models.

0.C.1 Analysis of SEA

Effect of reducing the radius. We complement the comparison of const-ϵ and red-ϵ schemes provided in Sec. 3.6 by showing the different robust mIoU achieved by the various algorithms. In Fig. 4 one can observe that, consistently with what reported for average pixel accuracy in Fig. 2, reducing the value of ϵ (red-ϵ APGD) outperforms in all cases the other schemes.

Analysis of individual components in SEA. To assess how much each loss contributes to the final performance of SEA, we report the individual performance (both accuracy and mIoU) at different ϵ_(∞) in Tab. 6, using robust models on Pascal-Voc and Ade20K. We recall that each loss is optimized with 300 iterations of red-ϵ APGD. A common trend across all models is that either ℒ_(MCE) or ℒ_(JS) are best individual attacks for accuracy whereas ℒ_(MCE-BAL) attacks the mIoU the best. Overall, SEA significantly reduces the worst case over individual attacks.

Analysing attack pairs in SEA. Further insights into SEA are given by looking at how different pairs of the components of SEA perform. Tab. 7 presents such evaluation for the robust UPerNet on Pascal-Voc from Tab. 1: as expected, MCE + JS yields the best robust aAcc, while the pairs with MCE-Bal have the lowest mIoU. Moreover, the worst-case over all losses (SEA) gives further improvements.

| Loss pair | 4/255 Acc | 4/255 mIoU | 8/255 Acc | 8/255 mIoU | 12/255 Acc | 12/255 mIoU |
| --- | --- | --- | --- | --- | --- | --- |
| ℒ_(MCE)+ℒ_(MCE-Bal) | 88.8 | 65.1 | 73.2 | 35.1 | 31.6 | 5.6 |
| ℒ_(MCE)+ℒ_(JS) | 88.6 | 64.9 | 72.2 | 35.2 | 29.4 | 6.0 |
| ℒ_(JS)+ℒ_(MCE-Bal) | 88.8 | 64.9 | 73.0 | 34.7 | 32.6 | 5.6 |
| SEA | 88.6 | 64.9 | 71.7 | 34.6 | 28.1 | 5.5 |

Table 7: Effectiveness of pairs of losses. We evaluate by pairing subset of components of SEA by measuring Acc and mIoU. Different pairs perform better or worse depending on perturbation strengths, while SEA always yields the strongest attack.

[Refer to caption]

Figure 5: Influence of number of iterations in SEA. We show robust average pixel accuracy (left) and mIoU (right) varying the number of iterations in our attack: 300 iterations give the best compute-effectiveness trade-off. We use the 5 step PIR-AT Pascal-Voc trained ConvNeXt-T backbone UPerNet model and the attack is done for ℓ_(∞) = 8/255.

More iterations. We also explore the effect of different number of iterations in SEA. In Fig. 5 we show the performance (measured by robust accuracy and mIoU) of SEA with 50, 100, 200, 300 and 500 iterations. There is a substantial improvement going from 50 to 300 iterations in all cases. On further increasing the number of attack iterations to 500, the drop in robust accuracy and mIoU is around 0.1% for both ℓ_(∞) radii of 8/255 and 12/255. Since going beyond 300 iterations gives no or minimal improvement for significantly higher computational cost, we fix the number of iterations to 300 in SEA.

Effect of random seed. We study the impact of the randomness involved in our algorithm (via random starting points for each run) by repeating the evaluation on our robust model on Pascal-Voc with 3 random seeds. Tab. 8 shows that the proposed SEA is very stable across all perturbation strengths. It is also interesting to note that all individual losses have negligible variance across the different runs.

| Model | ϵ_(∞) | ℒ_(MCE) Acc | ℒ_(MCE) mIoU | ℒ_(MCE-Bal) Acc | ℒ_(MCE-Bal) mIoU | ℒ_(JS) Acc | ℒ_(JS) mIoU | SEA Acc | SEA mIoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UPerNet ConvNeXt-T, PIR-AT, 50 epochs | 4/255 | 89.2±0.2 | 65.8±0.3 | 90.4±0.1 | 69.0±0.2 | 88.7±0.1 | 64.9±0.4 | 88.6±0.1 | 64.9±0.4 |
| UPerNet ConvNeXt-T, PIR-AT, 50 epochs | 8/255 | 73.8±0.4 | 40.8±0.4 | 77.5±0.2 | 38.1±0.2 | 73.9±0.1 | 41.3±0.0 | 71.7±0.3 | 34.6±0.1 |
| UPerNet ConvNeXt-T, PIR-AT, 50 epochs | 12/255 | 31.5±0.3 | 10.2±0.2 | 36.9±0.2 | 6.6±0.1 | 38.6±0.4 | 15.0±0.1 | 28.1±0.2 | 5.5±0.3 |

Table 8: Stability of SEA across different runs. We report Acc computed on Pascal-Voc with the 5 step UPerNet model trained with PIR-AT. The mean across 3 runs is shown along with the standard deviation. Across components and perturbation strengths, SEA has a very low variance over random seeds.

0.C.2 Excluding the background class from evaluation

For Ade20K, we train clean UPerNet + ConvNeXt-T models in two settings, i.e. either ignoring the background class (150 possible classes), which is the standard practice while training clean semantic segmentation models, or to predict it (151 classes). To measure the effect of the additional background class, we can evaluate the performance of both models with only 150 classes (for the one trained on 151 classes, we can exclude the score of the background class when computing the predictions). Training on 150 classes achieves (Acc, mIoU) of (80.4%, 43.8%), compared to (80.2%, 43.8%) for 151. This shows that we do not lose any performance when training with the background class, and the lower clean accuracy of clean trained Ade20K models, (Acc, mIoU) of (75.5%, 41.1%) is due to including the background class when computing the statistics. This also translates to the robust models trained in the 2 step PIR-AT setting. For the robust model, the two settings have (76.6%, 37.8%) and (76.4%, 37.5%) (Acc, mIoU) respectively.

0.C.3 Additional comparisons to existing attacks

[Refer to caption]

Figure 6: Comparison to ALMA prox. We compare APGD with our novel loss (ℒ_(Mask-CE)) and the ensemble SEA according to the metric used by [36], which differs from those (Acc and mIoU) we use in the rest of our experiments. In the left plot, the attacks are tested on a clean trained model for the Pascal-Voc dataset, and in the right plot we test against our robust PIR-AT model.

Rony et al. [36] have recently proposed ALMA prox as an adversarial attack against semantic segmentation models: its goal is to reach, for each image, a fixed success rate threshold (i.e. a certain percentage of mis-classified pixels, in practice 99% is used) with a perturbation of minimal ℓ_(∞) norm. Thus, the threat model considered by [36] is not comparable to ours, which aims at reducing average pixel accuracy as much as possible with perturbations of a limited size.

In order to provide a comparison of our algorithms to ALMA prox, we measure the percentage of images for which the attack cannot make 99% of pixels be misclassified with perturbations of ℓ_(∞)-norm smaller than a threshold ϵ (i.e. the model is considered robust on such images). In this case, lower values indicate stronger attacks. We show in Fig. 6 the results in such metric, at various ϵ, for ALMA prox (default values, 500 iterations), APGD on the Mask-CE loss (300 iterations) and SEA. We test for 160 random images from the Pascal-Voc dataset using the clean trained UPerNet with a ConvNeXt-T backbone in the left plot and 5-step adversarially trained version of the same model in the right plot Fig. 6. For the clean model (left plot) the three attacks perform similarly, with a slight advantage of SEA at most radii. However, on the robust model (right plot), both APGD on the Mask-CE loss and SEA significantly outperform ALMA prox: for example, APGD, which uses even less iterations than ALMA prox, attains 0% robustness at 32/255, compared to 77% of ALMA prox. This shows that, even considering a different threat model, our attacks are effective to estimate adversarial robustness.

0.C.4 Additional discussion of existing PGD-based attacks

Recently, [21, 1] revisited the loss used in the attack to improve the effectiveness of ℓ_(∞)-bounded attacks, and are closest in spirit to our work. Since these methods represent the main baseline for our attacks, in the following we briefly summarize their approach to highlight the novelty of our proposed losses.

SegPGD:

    [21] proposes to balance the importance of the cross-entropy loss of correctly and wrongly classified pixels over iterations. In particular, at iteration t = 1, …, T, they use, with λ(t) = (t−1)/(2T),

      -- --------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------- --
         ℒ_(SegPGD)(u,y) = (   ${{({1 - {\lambda{(t)}}})} \cdot {\mathbb{I}}}{({{\underset{j = {1,\ldots,K}}{\arg\max}{\mathbf{u}}_{j}} = y})}$
                               $+ \lambda{(t)} \cdot {\mathbb{I}}{(\underset{j = {1,\ldots,K}}{\arg\max}{\mathbf{u}}_{j} \neq y)}) \cdot \mathcal{L}_{\text{CE}}({\mathbf{u}},y).$
      -- --------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------- --

    In this way the algorithm first focuses only on the correctly classified pixels and then progressively balances the attention given to the two subset of pixels: this has the goal of avoiding to make updates which find new misclassified pixels but leads to correct decisions for already misclassified pixels.

CosPGD:

    [1] proposes to weigh the importance of the pixels via cosine similarity between the prediction vector (after applying the sigmoid function σ(t) = 1/(1+e^(−t))) and the one-hot encoding e_(y) of the ground truth class. This can be written as

      -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --
         ${{\mathcal{L}_{\text{CosPGD}}{({\mathbf{u}},y)}} = {{\frac{\left\langle {\sigma{({\mathbf{u}})}},{\mathbf{e}}_{y} \right\rangle}{\left\| {\sigma{({\mathbf{u}})}} \right\|_{2}\left\| {\mathbf{e}}_{y} \right\|_{2}} \cdot \mathcal{L}_{\text{CE}}}{({\mathbf{u}},y)}} = {{{{\sigma{({\mathbf{u}}_{y})}}/\left\| {\sigma{({\mathbf{u}})}} \right\|_{2}} \cdot \mathcal{L}_{\text{CE}}}{({\mathbf{u}},y)}}},$
      -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --

    and again has the effect of reducing the importance of the pixels which are confidently misclassified.

0.C.5 Transfer attacks

| Dataset / Model | Attack | Source | Target | 0 Acc | 0 mIoU | 4/255 Acc | 4/255 mIoU | 8/255 Acc | 8/255 mIoU | 12/255 Acc | 12/255 mIoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ade20K, Segmenter with ViT-S backbone | APGD w/ ℒ_(Mask-CE) | clean | PIR-AT | 69.1 | 28.7 | 68.8 | 28.3 | 68.6 | 28.0 | 68.3 | 27.8 |
| Ade20K, Segmenter with ViT-S backbone | APGD w/ ℒ_(Mask-CE) | AT | PIR-AT | 69.1 | 28.7 | 66.3 | 26.0 | 63.1 | 23.8 | 57.4 | 19.9 |
| Ade20K, Segmenter with ViT-S backbone | SEA (white-box) | PIR-AT | PIR-AT | 69.1 | 28.7 | 55.3 | 14.9 | 33.3 | 5.4 | 8.9 | 1.1 |

Table 9: Transfer attacks. We show the robustness of PIR-AT to various transfer attacks (measured with Acc and mIoU at various radii). For each case we indicate the source and target models. Moreover, we report the evaluation given by white-box attacks as baseline.

To complement the evaluation of the robustness of our PIR-AT models, we further test them with transfer attacks from less robust models. In particular, we run APGD on the Masked-CE loss on Segmenter models obtained with either clean training or AT (5 steps) on Ade20K. We then transfer the found perturbations to our PIR-AT (5 steps, 128 epochs), and report robust accuracy and mIoU in Tab. 9, together with the results of the white-box SEA on the same model (from Tab. 2) as baseline. We observe that the transfer attacks are far from the performance of SEA, which further supports the robustness of the PIR-AT models.

## Appendix 0.D Additional Figures

Untargeted attacks. Fig. 7 shows examples of our untargeted attacks at different radii ϵ_(∞) on the clean model for Pascal-Voc dataset. In particular, we use 300 iterations of red-ϵ APGD on the ℒ_(Mask-CE) loss. The first column presents the original image with the ground truth segmentation mask, The following columns contain the perturbed images and relative predicted segmentation masks for increasing radii (ϵ_(∞) = 0 is equivalent to the unperturbed image): one can observe that the model predictions progressively become farther away from the ground truth values. We additionally report the average pixel accuracy for each image. In Fig. 8, we repeat the same visualization for the most robust 5 step 300 epochs PIR-AT model. Note that we use different values of ϵ_(∞) for the two models, i.e. significantly smaller ones for the clean model, following Tab. 1. Finally, the same setup is employed on the UPerNet + ConvNeXt-T model trained for Ade20K dataset for the illustrations in Fig. 9 (clean model) and Fig. 10 (5-step robust PIR-AT model), and we have similar observations as for the smaller dataset. Again we use smaller radii for the clean model, since it is significantly less robust than the PIR-AT one.

Targeted attacks. In Fig. 1 we show examples of the perturbed images and corresponding predictions resulting from targeted attacks. In this case, we run APGD (red-ϵ scheme with 300 iterations) on the negative JS divergence between the model predictions and the one-hot encoding of the target class. In this way the algorithm optimizes the adversarial perturbation to have all pixels classified in the target class (e.g. “grass” or “sky” in Fig. 1). We note that other losses like cross-entropy can be adapted to obtain a targeted version of SEA, and we leave the exploration of this aspect of our attacks to future work.

- Figure 7, example A Acc by radius: `0 -> 95.9%`, `0.25/255 -> 94.8%`, `0.5/255 -> 75.9%`, `1/255 -> 48.3%`, `2/255 -> 0.0%`.

- Figure 7, example B Acc by radius: `0 -> 96.1%`, `0.25/255 -> 61.4%`, `0.5/255 -> 0.0%`, `1/255 -> 0.0%`, `2/255 -> 0.0%`.

Figure 7: Visualizing the perturbed images, corresponding predicted masks and Acc for increasing radii. The attacks are generated on the clean model on Pascal-Voc with APGD on ℒ_(Mask-CE). Original image and ground truth mask in the first column.

- Figure 8, example A Acc by radius: `0 -> 95.5%`, `4/255 -> 94.6%`, `8/255 -> 90.8%`, `12/255 -> 49.2%`, `16/255 -> 0.0%`.

- Figure 8, example B Acc by radius: `0 -> 93.7%`, `4/255 -> 92.7%`, `8/255 -> 83.3%`, `12/255 -> 6.8%`, `16/255 -> 0.0%`.

Figure 8: Same setting as in Fig. 7 for the 5-step PIR-AT model

- Figure 9, example A Acc by radius: `0 -> 65.9%`, `0.25/255 -> 54.9%`, `0.5/255 -> 4.9%`, `1/255 -> 0.0%`, `2/255 -> 0.0%`.

- Figure 9, example B Acc by radius: `0 -> 81.2%`, `0.25/255 -> 47.9%`, `0.5/255 -> 21.9%`, `1/255 -> 2.6%`, `2/255 -> 0.0%`.

Figure 9: Visualizing the perturbed images, corresponding predicted masks and Acc for increasing radii. The attacks are generated on the clean model on Ade20K with APGD on ℒ_(Mask-CE). Original image and ground truth mask in the first column.

- Figure 10, example A Acc by radius: `0 -> 61.3%`, `4/255 -> 58.6%`, `8/255 -> 29.7%`, `12/255 -> 1.6%`, `16/255 -> 0.0%`.

- Figure 10, example B Acc by radius: `0 -> 84.4%`, `4/255 -> 67.3%`, `8/255 -> 32.8%`, `12/255 -> 6.0%`, `16/255 -> 0.0%`.

Figure 10: Same setting as in Fig. 9 for the 5 step PIR-AT model.
