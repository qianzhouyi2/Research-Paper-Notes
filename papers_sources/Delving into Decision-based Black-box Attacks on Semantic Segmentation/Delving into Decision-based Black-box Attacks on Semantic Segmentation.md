# Delving into Decision-based Black-box Attacks on Semantic Segmentation

Zhaoyu Chen1,2* Zhengyang Shan1* Jingwen Chang1 Kaixun Jiang1 

Dingkang Yang1 Yiting Cheng2 Wenqiang Zhang1,2 

1 Shanghai Engineering Research Center of AI & Robotics, Academy for Engineering & Technology, Fudan University 2 Shanghai Key Lab of Intelligent Information Processing, School of Computer Science, Fudan University 

zhaoyuchen20@fudan.edu.cn 

# Abstract

Semantic segmentation is a fundamental visual task that finds extensive deployment in applications with securitysensitive considerations. Nonetheless, recent work illustrates the adversarial vulnerability of semantic segmentation models to white-box attacks. However, its adversarial robustness against black-box attacks has not been fully explored. In this paper, we present the first exploration of black-box decision-based attacks on semantic segmentation. First, we analyze the challenges that semantic segmentation brings to decision-based attacks through the case study. Then, to address these challenges, we first propose a decision-based attack on semantic segmentation, called Discrete Linear Attack (DLA). Based on random search and proxy index, we utilize the discrete linear noises for perturbation exploration and calibration to achieve efficient attack efficiency. We conduct adversarial robustness evaluation on 5 models from Cityscapes and ADE20K under 8 attacks. DLA shows its formidable power on Cityscapes by dramatically reducing PSPNet’s mIoU from an impressive $7 7 . 8 3 \%$ to a mere $2 . 1 4 \%$ with just 50 queries. 

# 1. Introduction

Deep neural networks (DNNs) have made unprecedented advancements and are extensively employed in various fundamental vision tasks, such as semantic segmentation [5, 25, 35] and video object segmentation [17–19]. However, recent studies have revealed the susceptibility of DNNs to adversarial examples [6, 7, 9, 31] by adding specially designed small perturbations to the input that are imperceptible to humans. The emergence of adversarial examples has prompted researchers to focus on the security of underlying visual tasks and seek inspiration for the development of robust DNNs through the exploration of adversarial examples. 

Semantic segmentation is a primary visual task for pixellevel classification. Despite its extensive utilization in realworld safety-critical applications like autonomous driving and medical image segmentation, it remains susceptible to adversarial examples. Recently, the emergence of Segment Anything Model (SAM) [22] has attracted people’s attention to segmentation models and inspired exploration of their robustness. However, there are few adversarial attacks on semantic segmentation [15], and they focus more on white-box attacks. White-box attacks require access to all information about the model (e.g., gradients and network architecture), which is challenging and often unattainable in real-world scenarios. Consequently, black-box attacks offer a more effective means to explore the adversarial robustness of semantic segmentation models in real-world scenarios. 

In this paper, we explore for the first time black-box attacks on semantic segmentation in the decision-based setting. The decision-based setting represents the most formidable challenge among black-box attacks, as it restricts access solely to the output category provided by the target model, without any information regarding probabilities or confidences. Nevertheless, the efficacy of decisionbased attacks on semantic segmentation remains severely constrained by the inherent characteristics of pixel-level classification, as evidenced by the following observations: i) Inconsistent Optimization Goal. In image classification, decision-based attacks often reduce the magnitude of perturbations under the premise of misclassification. However, in semantic segmentation, the larger the perturbation amplitude, the lower the metric, and it is difficult to constrain the perturbation to the $l _ { p }$ norm. ii) Perturbation Interaction. Perturbations from different iterations interfere with each other, so a pixel is classified incorrectly in this iteration but may be classified correctly under perturbation in the next iteration, which leads to optimization difficulties. iii) Complex Parameter Space. Attacking semantic segmentation is a multi-constraint optimization problem, wherein the complexity of the parameter space imposes lim-

itations on attack efficiency. In practice, it becomes imperative to employ an efficient decision-based black-box attack to assess the adversarial robustness of semantic segmentation. Therefore, the proposed attack must exhibit both high attack efficiency and reliable attack performance. 

To tackle the aforementioned challenges, we first propose the decision-based attack on semantic segmentation, termed Discrete Linear Attack (DLA). DLA employs a random search framework to effectively generate adversarial examples from clean images, utilizing a proxy index to guide the optimization process. Specifically, we optimize the adversarial examples by leveraging the changes in the proxy index corresponding to the image. Additionally, we alleviate the challenges identified in Section 3.2 by proposing discrete linear noises for updating the adversarial perturbation. For interference between perturbations, we find that locally adding noises has a good attack effect, but the added colorful patches are easily perceived. Therefore, we introduce linear noises and update the perturbation by adding horizontal or vertical linear noises to the image. To further compress the parameter space, we convert the complex continuous parameter space into a discrete parameter space and bisect the discrete noise from the extreme point of the $l _ { \infty }$ -norm ball. The overall process can be divided into two parts: perturbation exploration and perturbation calibration. In perturbation exploration, DLA adds discrete linear noises to the input to obtain a better initialization. In the perturbation calibration, DLA adaptively flips the perturbation direction of some regions according to the proxy index, updates and calibrates the perturbation. We evaluate the adversarial robustness of semantic segmentation models based on convolutional neural networks (FCN [25], PSPNet [35], DeepLabv3 [5]) and transformer (SegFormer [34] and Maskformer [10]) on Cityscapes [11] and ADE20K [36]. Extensive experiments demonstrate that DLA achieves state-of-the-art attack efficiency and performance on semantic segmentation. Our main contributions and experiments are as follows: 

• We first explore the adversarial robustness of existing semantic segmentation models based on decisionbased black-box attacks, including CNN-based and transformer-based models. 

• We analyze and summarize the challenges of decisionbased attacks on semantic segmentation. 

• We first propose the decision-based attack on semantic segmentation, called Discrete Linear Attack (DLA), which applies discrete linear noises to perturbation exploration and perturbation calibration. 

• Extensive experiments show the adversarial vulnerability of existing semantic segmentation models. On Cityscapes, DLA can reduce PSPNet’s mIoU from $7 7 . 8 3 \%$ to $2 . 1 4 \%$ within 50 queries. 

# 2. Related Work

Semantic Segmentation. Semantic segmentation is a visual task of pixel-level classification. Currently, DNNbased methods have become the dominant way of semantic segmentation since the seminal work of Fully Convolutional Networks (FCNs) [25]. The subsequent model focuses on aggregating long-range dependencies in the final feature map: DeepLabv3 [5] uses atrous convolutions with various atrous rates and PSPNet [35] applies pooling technology with different kernel sizes. The subsequent work began to introduce transformers [32] to model context: Seg-Former [34] replaces convolutional backbones with Vision Transformers (ViT) [12] that capture long-range context starting from the very first layer. MaskFormer [10] introduces the mask classification and employs a Transformer decoder to compute the class and mask prediction. 

Black-box Adversarial Attack. In this paper, we primarily concentrate on query-based black-box attacks, where it is assumed that attackers have limited access to the target network and can only make queries to obtain the network’s outputs (confidences or labels) for specific inputs [8, 23]. The former are called score-based attacks, while the latter are decision-based attacks. Generally speaking, scorebased attacks have higher attack efficiency on image classification. For decision-based attacks on semantic segmentation, we define the model output as the label of each pixel. Considering that the mIoU of semantic segmentation is a continuous value calculated based on the label of each pixel, we choose score-based attacks on image classification as the competitors in this paper. Most score-based attacks on image classification estimate the approximate gradient through zeroth-order optimizations [20]. Bandits [21] further introduce the gradient prior information and bandits to accelerate [20]. Then, Liu et al. [24] introduce the zeroth-order setup to sign-based stochastic gradient descent (SignSGD) [3] and propose ZO-SignSGD [24]. Then, Sign-Hunter [1] exploits the separability property of the directional derivative and improves the query efficiency. Recently, methods based on random search have been proposed and have better query efficiency. SimBA [16] randomly samples a vector from a predefined orthonormal basis to images. Square Attack [2] selects localized squareshaped updates at random positions to update perturbations. Compared with previous work, DLA analyzes the challenges of semantic segmentation and implements queryefficient attacks based on discrete linear noise. 

Adversarial Attack on Semantic Segmentation. Compared to image classification, there are few adversarial attacks on semantic segmentation. [14] and [33] are the first to study the adversarial robustness of semantic segmentation and illustrate its vulnerability through extensive experiments. Indirect Local Attack [28] reveals the adversarial vulnerability of semantic segmentation models due to 

![image](images/56c121918580b32a5b9650c62e17b72aa5856efbc7ed50e3c392930469eacd57.jpg)


![image](images/0e6b605e54ea9c7ab27b927996d774ebf403c253e860f30b1c5e5097ce26e4e4.jpg)



Figure 1. Based on Random attack, we give the changes in mIoU under various perturbation magnitudes. If we add a very large perturbation, this can make the mIoU very small. However, when reducing the perturbation magnitude, the mIoU increases, which makes the optimization goal and attack direction inconsistent.


long-range context. SegPGD [15] improves white-box attacks from the perspective of loss functions and can better evaluate and boost segmentation robustness. ALMA prox [29] produces adversarial perturbations with much smaller $l _ { \infty }$ norms with a proximal splitting. The aforementioned attacks primarily prioritize enhancing the strength of white-box attacks on semantic segmentation, while allocating comparatively less emphasis on the adversarial robustness of query-based black-box attacks. Consequently, as a complementary approach, we undertake the pioneering exploration of adversarial robustness within the highly challenging decision-based setting. 

# 3. Method

# 3.1. Preliminaries

In semantic segmentation, given the semantic segmentation model f(·), the clean image is x ∈ [0, 1]C×H×W $f ( \cdot )$ $x \in [ 0 , 1 ] ^ { C \times H \times \mathbf { \bar { W } } }$ and the corresponding labels are $y _ { i } \in \{ 1 , . . . , K \} ^ { d }$ $\mathit { \Pi } _ { M } = H W$ and $i = 1 , . . . , d )$ , where $C$ is the number of channels, $H$ and $W$ are the height and width of the image, and $K$ is the number of semantic classes. We denote the adversarial example $x _ { a d v } = x + \delta$ , where $\delta ^ { C \times H \times W }$ is the adversarial perturbation and it satisfies $| | \delta | | _ { \infty } \leq \epsilon$ . Because the attack is the decision-based setting, we denote the model output as the per-pixel predicted labels $\hat { y } ~ = ~ f ( x ) ~ \in ~ \{ 1 , . . . , K \} ^ { d }$ . We hope that the adversarial example can make all pixels misclassified as much as possible, so the optimization goal can be expressed as: 

$$
\underset {\delta} {\arg \max } \sum 1 (f (x + \delta) _ {i} \neq y _ {i}), \tag {1}
$$

$$
\begin{array}{l} \text {s . t .} | | \delta | | _ {\infty} \leq \epsilon \text {a n d} i = 1, \dots , d, \end{array}
$$

where 1 is the indicator function. When the condition is met, it is recorded as 1, otherwise it is 0. 

![image](images/a8b6dcee27097e772217bd06f146624c2ca0218686e606abc27ee22a9e14ce73.jpg)


![image](images/daff0f5e05bd11d857b49d9dcdf703ac7a4fbd7d6aae93c3f0f76daa40c49d6e.jpg)



Figure 2. Random attack with different proxy indexes. Our design focuses on optimizing the adversarial perturbation by initiating from clean images and iteratively updating the example based on the observed changes in the proxy index.


![image](images/2a1a04523d7179824ffe236118f2bfada48757ef7ffacf77d73df9db9770e7cc.jpg)



Figure 3. When facing black-box attacks on semantic segmentation, the update of perturbations causes the attacked pixels to revert to their original categories, resulting in optimization difficulties.


# 3.2. Attack Analysis

Decision-based attacks on image classification have been extensively and intensively studied [23], however, semantic segmentation has not been fully explored. Semantic segmentation is a pixel-level classification, which is far more difficult to attack than image classification, because attacking image classification is a single-constraint optimization, while every pixel in semantic segmentation must be classified, which results in attacking semantic segmentation as a multi-constraint optimization. Consequently, decisionbased attacks on semantic segmentation encounter substantial challenges, often leading to optimization convergence towards local optima, as illustrated below. 

Inconsistent Optimization Goal. Decision-based attacks on image classification commonly rely on boundary attacks [23]. Boundary attack [23] requires that the image is classified incorrectly, and then minimizes perturbations’ magnitude so that the adversarial example is near the decision boundary. However, this strategy cannot be applied in semantic segmentation. As shown in Figure 1, we can add very large noise to make the mean Intersection-over Union (mIoU) very small, but when reducing the noise magnitude, unlike image classification that maintains misclassification, the mIoU also becomes greater, which makes the optimization goal and attack direction inconsistent. 


Clean


![image](images/2dd9ac80bb3dd1766bf9358ebb45069429e06a789d582a6bbeb292ba983a5c23.jpg)



Random


![image](images/2539e28682f3f3753918ea7ebab6406a7aaec1c596fd895837aad32cc4e007dd.jpg)



Patch with Overlap


![image](images/8c83ad19d245251977a0e26fc049b8ae45e19cab379708cb2dd372bece0ec4fd.jpg)



Patch without Overlap


![image](images/7b0c9b3b84280fb59148bc319b903d2c0001caf7527802a77b504faf204cf86a.jpg)



Line


![image](images/8cfa7bcae66940cbe68b99b2246b71175fea3914480913879a54ad2d9bea9be9.jpg)


![image](images/4f816a08ee0cd61e83963b3836cc7edb68e3665c5aa1ec07f624acd69e6a2a13.jpg)


![image](images/b0b818742775169ba8eeb770bd51b33d14d90e9ac4b152ac3f001f3c5b5ded8c.jpg)


![image](images/dbf73b8f3b4095d8dae8be171725555f6acf0345a2eba5d213bba21a16fe405f.jpg)


![image](images/324ef2668faa674180955e69dd314496d9e7f9f07ba9dadcaf2e3921f59ad1b2.jpg)


![image](images/7bb3bcbfd53a465289febfc8758490915f54f9bdea1cdb931713955b9b84437e.jpg)



Figure 4. Description of Perturbation Interaction. We use perturbations in the form of random, patch with overlap, patch without overlap, and line to attack, which shows that there is interference between perturbations. Less overlap can lead to better attack performance and linear noises achieve better results in both imperceptibility and attack.


To address this challenge, we propose the utilization of a proxy index to generate adversarial examples from clean images. Our approach involves optimizing the adversarial perturbation starting from clean images and updating the example based on the changes observed in the proxy index associated with the image. To gain a deeper understanding of the proxy index, we propose a simple baseline method called Random Attack. The update process for this baseline method is as follows: 

$$
x _ {a d v} ^ {0} = x, x _ {a d v} ^ {t + 1} = \Pi_ {\epsilon} \left(x _ {a d v} ^ {t} + r a n d \left[ - \frac {\epsilon}{1 6}, + \frac {\epsilon}{1 6} \right]\right), \tag {2}
$$

where $r a n d [ - \frac { \epsilon } { 1 6 } , + \frac { \epsilon } { 1 6 } ]$ generates a noise that is the same as the input’s dimension and satisfies the random distribution $[ - \frac { \epsilon } { 1 6 } , + \frac { \epsilon } { 1 6 } ]$ , and $\Pi _ { \epsilon }$ clips the input to $[ x - \epsilon , x + \epsilon ]$ . During the iteration, Random Attack updates the perturbation only when the proxy index becomes smaller. The complete algorithm of Random Attack is in Supplementary Material B. 

Building upon Random Attack, we conduct a toy study using PSPNet [35] and SegFormer [34] on Cityscapes [11] and ADE20K [36], following the same experimental settings as described in Section 4. Considering that mIoU is a widely adopted metric [13] for evaluating semantic segmentation, it holds potential as a suitable proxy index. Additionally, the per-pixel classification accuracy (PAcc) can also reflect the attack performance. Hence, we select mIoU and PAcc as the proxy indices, and the attack process using Random Attack is illustrated in Figure 2. Our observations are as follows: i) Random Attack based on the proxy index can reduce the mIoU of the image, ii) when mIoU is employed as the proxy index, the attack performance is superior. This is because when PAcc is used as the proxy index, the adversarial example only needs to maximize misclassification at each pixel, without considering the overall class. Conversely, when mIoU is used as the proxy index, the mIoU of an individual image approaches the mIoU of the entire dataset, resulting in improved attack performance. Therefore, we select mIoU as the proxy index for our study. 

Perturbation Interaction. Despite the effectiveness of Random Attack with the proxy index, as depicted in 

Figure 2, we observe that its attack performance is constrained and prone to convergence, suggesting that it may have reached a local optimal solution. Recent research [15] demonstrates that during white-box attacks on semantic segmentation, the classification of each pixel exhibits instability. In one iteration, a pixel may be misclassified, while in the next iteration, it could be classified correctly. This situation also occurs in black-box attacks on semantic segmentation, as shown in Figure 3. 

Upon revisiting Random Attack, we hypothesize that there exists interference between the perturbations added in each iteration. This interference arises due to the inconsistent update direction between black-box and white-box attacks. Consequently, a pixel may succeed in one iteration of the attack but fail in the subsequent iteration, leading to convergence towards a local optimal solution. To mitigate this issue, we propose updating the perturbation not on the entire image but on a local region. This localized perturbation update approach may alleviate the interference and enhance the attack performance. 

Taking inspiration from this observation, we explore different perturbation update strategies of varying shapes and conduct corresponding experiments. The visualization of these strategies is presented in Figure 4. When random perturbations are added to the entire image, the resulting segmented mask generally remains close to the original prediction, and the object’s outline is relatively well-preserved. This aligns with the limited attack performance depicted in Figure 2. However, when we update the perturbation in the form of patches with overlap [2], we observe minimal changes in the attack performance, and the object’s outline is still well-maintained. Conversely, when the perturbations are patches without overlap [21], a significant portion of the object’s outline is destroyed, indicating the presence of interference between perturbation updates. 

Looking back at the adversarial example in patches without overlap, although its attack effect is significant, it is easy to observe that carefully designed perturbations are added because the added patches are blocky and the color is ob-


Table 1. Search strategy for perturbation values. We report the mIoU under 50/200 query budgets and observe that for the same queries, discrete perturbations always obtain lower mIoU $( \% )$ .


<table><tr><td rowspan="2">Datset</td><td rowspan="2">Model</td><td colspan="4">Attack</td></tr><tr><td>Clean</td><td>Random</td><td>NES</td><td>Discrete</td></tr><tr><td rowspan="2">Cityscapes</td><td>PSPNet</td><td>77.83</td><td>48.81/47.18</td><td>48.34/47.40</td><td>33.57/33.54</td></tr><tr><td>SegFormer</td><td>80.43</td><td>58.59/56.07</td><td>58.00/55.34</td><td>41.70/41.70</td></tr><tr><td rowspan="2">ADE20K</td><td>PSPNet</td><td>37.68</td><td>26.63/26.34</td><td>25.67/25.52</td><td>23.50/23.31</td></tr><tr><td>SegFormer</td><td>43.74</td><td>34.72/34.45</td><td>34.66/34.55</td><td>33.68/33.53</td></tr></table>

vious. To ensure an effective attack and the perturbation is imperceptible simultaneously, we consider modeling the form of the perturbation as a line, as shown in Figure 4. We primarily choose linear noises for the following reasons: i) local adversarial perturbations can be spread to the global through context modeling of semantic segmentation [28], thereby attacking pixels in other areas, so linear noises are still effective. ii) Linear noises are thinner compared to patches, making them relatively harder to detect by the human eye. As depicted in Figure 4, linear noises exhibit superior performance compared to other strategies while remaining imperceptible. 

Complex Parameter Space. Despite the effectiveness of linear noises in enhancing attack performance, the presence of complex parameter spaces still hampers attack efficiency. Semantic segmentation poses a multi-constraint optimization problem, making it challenging to find the optimal adversarial example within a limited query budget. 

In black-box attacks, we usually use two methods to update the perturbation value: random noise [2, 16] and gradient estimation [20, 21]. Random noise causes clean images to randomly walk on the decision boundary and hope to cross it. Gradient estimation is a gradient-free optimization technology that approximates a gradient direction through random sampling, which can speed up attack efficiency and the most commonly used one is Natural Evolutionary Strategies (NES) [20]. Although both of the above strategies are effective, they still require many queries, and the query budget increases significantly as the parameter space becomes larger [21]. Even if [21] introduces prior information to reduce the parameter space, the query efficiency is relatively limited. Therefore, we consider further reducing the parameter space. 

For limited queries, it is unlikely to enumerate the entire parameter space. Recent work [4, 27] shows that adversarial examples are often generated at the extreme points of $l _ { \infty }$ norm ball, which illustrates that it is easier to find adversarial examples at these extreme points. Empirical findings in [27] also suggest that adversarial examples obtained from PGD attacks [26] are mostly found on the extreme points of $l _ { \infty }$ norm ball. Inspired by this, we directly restrict the possible perturbation as the extreme points of the $l _ { \infty }$ norm ball and change the parameter space from continuous space to 


Algorithm 1 Discrete Linear Attack (DLA)


Input: the image $x$ , model $f$ , proxy index $L$ , iteration $T$ Output: $x_{adv}$ 1: $l_{\min} \gets L(x)$ , $\hat{\delta} \gets 0$ , $i \gets 0$ , $M \gets 1$ , $n \gets 0$ 2: for $t \in [1, T]$ do  
3: if $t \leq \frac{T}{5}$ then  
4: // Perturbation Exploration  
5: $k \gets t \% 2$ 6: $\delta \sim k \cdot \{-\epsilon, \epsilon\}^h + (1 - k) \cdot \{-\epsilon, \epsilon\}^w$ 7: if $l_{\min} > L(x + \delta)$ then  
8: $l_{\min} \gets L(x + \delta)$ , $\hat{\delta} \gets \delta$ , $d \gets k$ 9: end if  
10: else  
11: // Perturbation Calibration  
12: $c \gets d \cdot \left[\frac{h}{2^n}\right] + (1 - d) \cdot \left[\frac{w}{2^n}\right]$ 13: $M[d \times i \times c : d \times (i + 1) \times c + (1 - d) \times h, (1 - d) \times i \times c : (1 - d) \times (i + 1) \times c + d \times w] * = -1$ 14: if $l_{\min} > L(x + \hat{\delta} \cdot M)$ then  
15: $l_{\min} \gets L(x + \hat{\delta} \cdot M)$ , $\hat{M} \gets M$ 16: else  
17: $M[d \times i \times c : d \times (i + 1) \times c + (1 - d) \times h, (1 - d) \times i \times c : (1 - d) \times (i + 1) \times c + d \times w] * = -1$ 18: end if  
19: $i \gets i + 1$ 20: if $i == 2^n$ then  
21: $i \gets 0$ , $n \gets n + 1$ 22: end if  
23: if $n == [\log_2(d \cdot h + (1 - d) \cdot w)] + 1$ then  
24: $\hat{\delta} \gets \hat{\delta} \cdot M$ , $i \gets 0$ , $n \gets 0$ 25: end if  
26: end if  
27: end for  
28: $x_{adv} \gets x + \hat{\delta} \cdot M$ 29: return $x_{adv}$ 

discrete space. Specifically, the adversarial perturbation $\delta$ is sampled from the Binomial distribution $\{ - \epsilon , \epsilon \} ^ { d }$ , called discrete noises. In this way, we directly reduce the parameter space from $[ - \epsilon , \epsilon ] ^ { d }$ to $\{ - \epsilon , \epsilon \} ^ { d }$ , which only $2 ^ { d }$ possible search directions. We conduct a case study to illustrate the effectiveness of these discrete noises, as shown in Table 1. Here, we use Random Attack as the baseline and report the mIoU under 50 and 200 query budget. We observe that for the same number of queries, discrete noises can always obtain lower mIoU, and there is a significant gap with other strategies, which illustrates the effectiveness of reducing the parameter space. 

# 3.3. Discrete Linear Attack

In this section, we introduce the proposed Discrete Linear Attack (DLA) based on the aforementioned analysis. DLA consists of two main components: perturbation ex-

ploration and perturbation calibration. In the perturbation exploration phase, DLA introduces discrete perturbations in the horizontal or vertical direction to the input, aiming to achieve a better initialization. In the perturbation calibration phase, DLA dynamically flips the perturbation direction in certain regions based on the proxy index. This allows for iterative updates and calibration of the perturbation. The pipeline of DLA is as outlined in Algorithm 1. 

Perturbation Exploration. In Section 3.2, discrete linear noises can greatly compress the parameter space and improve attack efficiency. Combined with the proxy index and considering the aspect ratio of the image, we initialize the perturbation as follows: 

$$
x _ {a d v} \leftarrow x + \delta , \quad \delta \sim \{- \epsilon , \epsilon \} ^ {d}, \tag {3}
$$

where $d$ denotes the height or weight of images. In perturbation exploration, we alternately sample discrete linear noises with horizontal or vertical directions and add them to the clean image. Then, we calculate the proxy index and retain the adversarial perturbation that obtains the minimum proxy index as $\hat { \delta }$ . 

Perturbation Calibration. Although perturbation exploration has demonstrated high attack performance, the obtained adversarial perturbations still fall short of optimality. This limitation arises from the coarse-grained nature of perturbation exploration, which fails to consider the finegrained updating of local perturbations. Given the discrete nature of the noise, we propose generating new perturbations by flipping the sign of the existing perturbation. 

In the perturbation calibration phase, we adopt a hierarchical approach to randomly flip the sign of local perturbations, thereby further refining the perturbations. This process involves first attempting to flip the global perturbation and subsequently dividing the image into blocks, performing flipping operations on each block. Specifically, we first partition the entire image into blocks, then iterate over each block and flip the sign of the discrete linear perturbation. If the mIoU after flipping is lower, the current perturbation is saved. After traversing the current block, DLA further divides the image into more fine-grained blocks and then traverses. By employing hierarchical blocking and flipping, we aim to obtain the most effective adversarial examples. This operations are outlined in Lines 12-25 of Algorithm 1. 

# 4. Experiments

# 4.1. Experimental Setup

Datasets. We attack the semantic segmentation models with two widely used semantic segmentation datasets: Cityscapes [11] (19 classes) and ADE20K [36] (150 classes). Following [28] and [15], we randomly select 150 and 250 images from the validation set of Cityscapes and ADE20K. For evaluation metrics, we choose the standard 

metric, mean Intersection-over Union (mIoU) [13], a perpixel metric that directly corresponds to the per-pixel classification formulation. After attacking, the less the mIoU, the better the attack performance. 

Models. We select two types of semantic segmentation models: traditional convolutional models (FCN [25], DeepLabv3 [5], and PSPNet [35]), and transformer-based models (SegFormer [34] and MaskFormer [10]). Please refer to Supplementary Material $C$ for more model details. 

Attacks. We select 7 attack algorithms for performance comparison, including zero-order optimization (NES [20], Bandits [21], ZO-SignSGD [24], and SignHunter [1]) and random search (Random attack (Random), SimBA [16], and Square Attack [2] (Square)). 

Implementation details. In all experiments, the maximum perturbation epslion $\epsilon$ is 8. For NES [20], we set the number of queries for a single attack $q = 1 0$ . For Bandit Attack, we set the initial value of patch size priority $\iota _ { s i z e } \ = \ 2 0$ , and the learning rate priorexploration $= ~ 0 . 1$ . For ZO-SignSGD [24], we set the same number of single attack queries as NES $q \ = \ 1 0$ . For Square Attack [2], we set the initial value of the fraction of pixels $p _ { i n i t }$ is 0.05. For SimBA [16], we set the magnitude of the perturbation delta as 50. The setting of SignHunter [1] is consistent with the original paper. To alleviate the effect of randomness, we report average mIoU $( \% )$ after three attacks. 

# 4.2. Performance Comparison

Attack Results. Table 2 illustrates the attack results of 8 black-box attacks on Cityscapes [11] and ADE20K [36]. We report mIoU $( \% )$ of 5 models under 50 and 200 query budget. Random and NES [20] have lower attack performance due to their complex parameter spaces. ZO-SignSGD [24], SimBA [16], and Square [2] introduce local prior information, which can further improve attack performance. Furthermore, both Bandits [21] and Sign-Hunter [1] use non-overlapping local noise, thus achieving sub-optimal performance. However, as shown in Figure 5, Bandits’ patch noise and SignHunter’s strip noise are colored and are very easy to perceive by humans. Our DLA significantly outperforms other competing attacks on both datasets. On Cityscapes’ PSPNet, DLA reduces mIoU by $1 5 . 4 9 \%$ and $2 3 . 9 8 \%$ compared to Bandits and Signhunter under 200 queries. Further, on the more challenging PSPNet of ADE20K, DLA reduces mIoU by $1 4 . 0 8 \%$ and $1 0 . 3 1 \%$ compared to Bandits and Signhunter under 200 queries. In terms of visualization, our DLA maintains the imperceptibility of adversarial perturbations and is able to destroy the outline of objects well. In terms of attack efficiency, the attack performance of DLA under 50 queries exceeds the results of other attacks under 200 queries by a very significant gap. Overall, our DLA has extremely high attack efficiency and can more efficiently evaluate the adversarial 


Table 2. Attack results on Cityscapes and ADE20K. We report mIoU $( \% )$ ) under 50/200 query budget.


<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Attack</td><td colspan="5">Model</td></tr><tr><td>FCN [25]</td><td>PSPNet [35]</td><td>DeepLab V3 [5]</td><td>SegFormer [34]</td><td>MaskFormer [10]</td></tr><tr><td rowspan="9">Cityscapes [11]</td><td>Clean</td><td>77.89</td><td>77.83</td><td>77.70</td><td>80.43</td><td>73.91</td></tr><tr><td>Random</td><td>35.76/34.94</td><td>48.81/47.18</td><td>54.57/52.77</td><td>58.59/56.07</td><td>39.09/39.06</td></tr><tr><td>NES [20]</td><td>34.47/33.82</td><td>48.34/47.40</td><td>54.32/52.99</td><td>58.00/55.34</td><td>51.94/52.56</td></tr><tr><td>Bandits [21]</td><td>18.17/15.65</td><td>20.81/17.55</td><td>29.85/26.73</td><td>39.43/36.14</td><td>26.94/26.88</td></tr><tr><td>ZO-SignSGD [24]</td><td>34.97/34.01</td><td>46.69/45.80</td><td>51.83/50.54</td><td>55.67/54.81</td><td>49.65/49.59</td></tr><tr><td>SignHunter [1]</td><td>23.88/21.67</td><td>33.52/26.04</td><td>44.24/35.93</td><td>41.38/34.18</td><td>47.05/27.06</td></tr><tr><td>SimBA [16]</td><td>33.74/29.58</td><td>46.27/40.22</td><td>54.67/50.17</td><td>54.17/52.67</td><td>33.52/32.71</td></tr><tr><td>Square [2]</td><td>35.47/35.99</td><td>48.47/49.18</td><td>54.45/56.23</td><td>56.71/52.18</td><td>50.87/49.84</td></tr><tr><td>Ours</td><td>3.18/3.07</td><td>2.14/2.06</td><td>1.79/1.71</td><td>18.12/17.78</td><td>2.79/2.78</td></tr><tr><td rowspan="9">ADE20K [36]</td><td>Clean</td><td>33.54</td><td>37.68</td><td>39.36</td><td>43.74</td><td>45.50</td></tr><tr><td>Random</td><td>22.85/22.13</td><td>27.72/27.36</td><td>25.82/24.81</td><td>38.02/37.64</td><td>25.37/24.06</td></tr><tr><td>NES [20]</td><td>24.47/23.96</td><td>26.57/26.26</td><td>23.83/23.41</td><td>36.35/36.06</td><td>34.78/34.55</td></tr><tr><td>Bandits [21]</td><td>25.10/23.67</td><td>25.02/23.93</td><td>27.52/26.36</td><td>36.32/35.03</td><td>26.14/26.91</td></tr><tr><td>ZO-SignSGD [24]</td><td>23.29/22.94</td><td>26.82/26.47</td><td>25.38/24.41</td><td>35.22/32.18</td><td>33.32/32.86</td></tr><tr><td>SignHunter [1]</td><td>20.15/16.72</td><td>24.21/20.16</td><td>25.40/20.48</td><td>32.56/28.22</td><td>28.78/16.78</td></tr><tr><td>SimBA [16]</td><td>24.20/21.49</td><td>26.36/22.92</td><td>25.56/22.13</td><td>36.70/34.81</td><td>35.62/33.18</td></tr><tr><td>Square [2]</td><td>23.94/22.90</td><td>26.87/25.89</td><td>27.70/26.46</td><td>35.43/34.76</td><td>26.29/26.41</td></tr><tr><td>Ours</td><td>8.18/7.97</td><td>10.19/9.85</td><td>11.34/10.67</td><td>28.91/27.85</td><td>12.14/12.14</td></tr></table>

![image](images/4ea1500bbce3a5d479e6cd813cfd7f28ecb2d50798c0aa8052be543abdb1be78.jpg)



Figure 5. Visualization of different attacks on Cityscapes and the threat model is SegFormer.


robustness of existing semantic segmentation models. 

Discussion. In Table 2, we observe that decision-based attacks on ADE20K [36] are more challenging than attacking Cityscapes [11]. We think the possible reason is that the category distribution of images in Cityscapes is relatively even, and they are all urban scenes with relatively high similarity and low complexity, so it is easier to attack. ADE20K has more categories and the differences between images are larger, so the attack is more difficult. In addition, we also find that SegFormer [34] demonstrates the best adversarial robustness under 8 attacks on both datasets, compared with the other 4 semantic segmentation models. This is because SegFormer is a transformer-based model, its main components are transformers, and the self-attention mechanism leads to higher adversarial robustness [8, 30], which is consistent with the description in SegFormer. Furthermore, it is worth noting that the backbone of MaskFormer has the structure of CNN, which implies that it does not exhibit a 

higher level of robustness compared to SegFormer. 

# 4.3. Diagnostic Experiment

To study the effect of our core designs, we conduct ablative studies on Cityscapes and ADE20K. We use Seg-Former [34] as the threat model and attack it under 50/200 query budget. 

Attack Design. We first study the attack design of DLA, as shown in Table 3. In perturbation exploration, random is the random noise of Random Attack, and horizontal and vertical is to add discrete linear noise horizontally and vertically respectively. iterative is to add discrete linear noise alternately horizontally and vertically. In perturbation calibration, random is the random update noise of Random attack, and flip is the update strategy of DLA’s filp perturbation sign. We observe that flip can achieve better attack performance than random in perturbation calibration. When the perturbation exploration is iterative, under 200 

![image](images/2319316fa6220752dee080062ec66bbe2114870201c03dd5e28fca592d071e16.jpg)


![image](images/caed958db330400d47e4b407ef87c407b631b6e7345513763e2ee1ce2a3ce7b3.jpg)



Figure 6. Attack performance of different black-box attacks under different perturbation magnitudes $\epsilon$ within a 200 query budget.


queries, it exceeds random $0 . 7 6 \%$ on Cityscapes and $0 . 8 3 \%$ on ADE20K. In perturbation exploration, discrete linear noise significantly surpasses random noise by a large advantage. vertical and iterative achieve the best performance on Cityscapes and ADE20K respectively. We find that the aspect ratio of Cityscapes is fixed, so resulting in vertical noises being more effective. The aspect ratio of ADE20K changes, so iterative has better attack performance, which means it is more generalizable when facing images of more scales. Considering the robustness of facing images with different aspect ratios, we choose iterative as the strategy for adding discrete linear noises. 

Perturbation Magnitude ϵ. To assess the impact of different perturbation magnitudes $\epsilon$ on attack performance, we select ϵ as 4, 8, and 16 for experiments on SegFormer. Figure 6 depicts the attack performance of different black-box attacks under different perturbation magnitudes ϵ. As the magnitude of perturbations increases, all attacks exhibit a greater decrease in overall mIoU. Notably, DLA consistently achieves the highest attack performance across all three perturbation magnitudes ϵ. Additionally, we observe that as the magnitude of perturbations increases, DLA outperforms other competing attacks in terms of the extent to which it can degrade mIoU. The above experiments illustrate that DLA has a stronger ability to evaluate the adversarial robustness of semantic segmentation under different perturbation magnitudes than other competing attacks. 

Limited Queries. Since a large number of queries leads to detection by the target system, we test the attack performance under extremely limited queries. To simulate a limited number of queries, we give 10 query budgets and evaluate the mIoU after the attack, as shown in Table 4. On Cityscapes, DLA demonstrated extreme attack efficiency, 


Table 3. Ablation study on the attack design of DLA.


<table><tr><td colspan="3">Pert. Explor.</td><td>Pert. Callbr.</td><td colspan="2">Dataset</td></tr><tr><td>random</td><td>horizontal</td><td>vertical</td><td>iterative</td><td>random</td><td>flip</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>80.43</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>43.74</td></tr><tr><td>✓</td><td></td><td></td><td></td><td>✓</td><td>58.59/56.07</td></tr><tr><td>✓</td><td></td><td></td><td></td><td></td><td>38.02/37.64</td></tr><tr><td></td><td>✓</td><td></td><td></td><td>✓</td><td>55.47/55.12</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>36.94/36.57</td></tr><tr><td></td><td>✓</td><td></td><td></td><td>✓</td><td>26.41/26.21</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>32.02/31.65</td></tr><tr><td></td><td></td><td>✓</td><td></td><td>✓</td><td>17.53/17.10</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>29.61/28.49</td></tr><tr><td></td><td></td><td></td><td>✓</td><td>✓</td><td>18.29/18.26</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>29.56/28.77</td></tr><tr><td></td><td></td><td></td><td>✓</td><td></td><td>18.12/17.78</td></tr></table>


Table 4. Attack results under limited queries (10 query budget).


<table><tr><td>Attack</td><td>Clean</td><td>Random</td><td>NES</td><td>Bandits</td><td>ZO-SignSGD</td><td>SignHunter</td><td>SimBA</td><td>Square</td><td>Ours</td></tr><tr><td>Cityscapes</td><td>80.43</td><td>59.22</td><td>59.26</td><td>41.20</td><td>56.23</td><td>47.90</td><td>54.62</td><td>57.22</td><td>18.89</td></tr><tr><td>ADE20K</td><td>43.74</td><td>37.82</td><td>38.20</td><td>35.71</td><td>37.82</td><td>35.95</td><td>37.57</td><td>37.64</td><td>30.85</td></tr></table>

reducing SegFormer’s mIoU of $6 1 . 5 4 \%$ within 10 queries and surpassing the second-best bandits attack of $2 2 . 3 1 \%$ by a significant margin. In the more challenging ADE20K, our DLA reduces the mIoU of SegFormer by $1 2 . 8 9 \%$ in 10 queries. Likewise, we beat the next best bandits attack by $4 . 8 6 \%$ . Combined with the attack results in Table 1, our DLA has attack performance that exceeds other current competitive attacks under both limited queries and a large number of queries. It indicates DLA can effectively evaluate the adversarial robustness of semantic segmentation in industrial and academic scenarios. 

# 5. Conclusions

In this paper, we provide the first in-depth study of decisionbased attacks on semantic segmentation. For the first time, we analyze the differences between semantic segmentation and image classification and study the three major challenges of corresponding decision-based attacks. Based on random search and proxy index, we discover discrete linear noise and propose a novel discrete linear attack (DLA). We conduct extensive experiments on 2 datasets and 5 models. Compared with 7 attack competitors, DLA has higher attack performance and query efficiency. On Cityscapes, DLA can reduce PSPNet’s mIoU from $7 7 . 8 3 \%$ to $2 . 1 4 \%$ within 50 queries. Therefore, DLA is expected to evaluate adversarial robustness in security-sensitive applications. 

Broader Impacts. Our proposed method exhibits exceptional attack performance and efficiency, thereby presenting a significant and concerning threat to the field of semantic segmentation. This threat becomes particularly pronounced when considering its potential deployment in security-sensitive applications, such as medical diagnosis and autonomous driving. Consequently, we anticipate that this threat will serve as a catalyst for the development of robust designs for semantic segmentation models and will also draw increased attention from the public. 

# References



[1] Abdullah Al-Dujaili and Una-May O’Reilly. Sign bits are all you need for black-box attacks. In International Conference on Learning Representations, 2019. 2, 6, 7 





[2] Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. Square attack: a query-efficient black-box adversarial attack via random search. In European conference on computer vision, pages 484–501. Springer, 2020. 2, 4, 5, 6, 7 





[3] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signsgd: Compressed optimisation for non-convex problems. In International Conference on Machine Learning, pages 560–569. PMLR, 2018. 2 





[4] Jinghui Chen, Dongruo Zhou, Jinfeng Yi, and Quanquan Gu. A frank-wolfe framework for efficient and effective adversarial attacks. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 3486–3494, 2020. 5 





[5] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017. 1, 2, 6, 7 





[6] Zhaoyu Chen, Bo Li, Shuang Wu, Jianghe Xu, Shouhong Ding, and Wenqiang Zhang. Shape matters: deformable patch attack. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IV, pages 529–548. Springer, 2022. 1 





[7] Zhaoyu Chen, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, and Wenqiang Zhang. Towards practical certifiable patch defense with vision transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15148–15158, 2022. 1 





[8] Zhaoyu Chen, Bo Li, Shuang Wu, Shouhong Ding, and Wenqiang Zhang. Query-efficient decision-based black-box patch attack. IEEE Transactions on Information Forensics and Security, 18:5522–5536, 2023. 2, 7 





[9] Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, and Wenqiang Zhang. Content-based unrestricted adversarial attack. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 1 





[10] Bowen Cheng, Alexander G. Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 17864–17875, 2021. 2, 6, 7 





[11] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 3213–3223, 2016. 2, 4, 6, 7 





[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 2 





[13] Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. International journal of computer vision, 111:98–136, 2015. 4, 6 





[14] Volker Fischer, Mummadi Chaithanya Kumar, Jan Hendrik Metzen, and Thomas Brox. Adversarial examples for semantic image segmentation. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop Track Proceedings. OpenReview.net, 2017. 2 





[15] Jindong Gu, Hengshuang Zhao, Volker Tresp, and Philip HS Torr. Segpgd: An effective and efficient adversarial attack for evaluating and boosting segmentation robustness. In European Conference on Computer Vision, pages 308–325. Springer, 2022. 1, 3, 4, 6 





[16] Chuan Guo, Jacob Gardner, Yurong You, Andrew Gordon Wilson, and Kilian Weinberger. Simple black-box adversarial attacks. In International Conference on Machine Learning, pages 2484–2493. PMLR, 2019. 2, 5, 6, 7 





[17] Pinxue Guo, Wei Zhang, Xiaoqiang Li, and Wenqiang Zhang. Adaptive online mutual learning bi-decoders for video object segmentation. IEEE Transactions on Image Processing, 31:7063–7077, 2022. 1 





[18] Lingyi Hong, Wei Zhang, Liangyu Chen, Wenqiang Zhang, and Jianping Fan. Adaptive selection of reference frames for video object segmentation. IEEE Transactions on Image Processing, 31:1057–1071, 2021. 





[19] Lingyi Hong, Wenchao Chen, Zhongying Liu, Wei Zhang, Pinxue Guo, Zhaoyu Chen, and Wenqiang Zhang. Lvos: A benchmark for long-term video object segmentation. arXiv preprint arXiv:2211.10181, 2022. 1 





[20] Andrew Ilyas, Logan Engstrom, Anish Athalye, and Jessy Lin. Black-box adversarial attacks with limited queries and information. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmassan, Stockholm, Sweden, July 10-15, 2018 ¨ , pages 2142–2151. PMLR, 2018. 2, 5, 6, 7 





[21] Andrew Ilyas, Logan Engstrom, and Aleksander Madry. Prior convictions: Black-box adversarial attacks with bandits and priors. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. 2, 4, 5, 6, 7 





[22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer ´ Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, ´ and Ross B. Girshick. Segment anything. CoRR, abs/2304.02643, 2023. 1 





[23] Huichen Li, Xiaojun Xu, Xiaolu Zhang, Shuang Yang, and Bo Li. QEBA: query-efficient boundary-based blackbox at-





tack. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 1218–1227, 2020. 2, 3 





[24] Sijia Liu, Pin-Yu Chen, Xiangyi Chen, and Mingyi Hong. signsgd via zeroth-order oracle. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. 2, 6, 7 





[25] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 3431–3440. IEEE Computer Society, 2015. 1, 2, 6, 7 





[26] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In ICLR, 2018. 5 





[27] Seungyong Moon, Gaon An, and Hyun Oh Song. Parsimonious black-box adversarial attacks via efficient combinatorial optimization. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, pages 4636–4645, 2019. 5 





[28] Krishna Kanth Nakka and Mathieu Salzmann. Indirect local attacks for context-aware semantic segmentation networks. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, pages 611–628. Springer, 2020. 2, 5, 6 





[29] Jer´ ome Rony, Jean-Christophe Pesquet, and Ismail ˆ Ben Ayed. Proximal splitting adversarial attack for semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20524–20533, 2023. 3 





[30] Rulin Shao, Zhouxing Shi, Jinfeng Yi, Pin-Yu Chen, and Cho-Jui Hsieh. On the adversarial robustness of vision transformers. Transactions on Machine Learning Research, 2022. 7 





[31] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013. 1 





[32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. 2 





[33] Cihang Xie, Jianyu Wang, Zhishuai Zhang, Yuyin Zhou, Lingxi Xie, and Alan L. Yuille. Adversarial examples for semantic segmentation and object detection. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, pages 1378–1387. IEEE Computer Society, 2017. 2 





[34] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34:12077–12090, 2021. 2, 4, 6, 7 





[35] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 6230–6239. IEEE Computer Society, 2017. 1, 2, 4, 6, 





[36] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ADE20K dataset. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 5122–5130. IEEE Computer Society, 2017. 2, 4, 6, 7 

