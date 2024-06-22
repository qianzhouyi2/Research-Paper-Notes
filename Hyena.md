![[hyena.assets/Pasted image 20240610195712.png]]
# 鬣(Liè)狗层次结构：迈向更大的卷积语言模型

## 摘要：
Transformer的注意力模块对于序列长度有二次成本，限制了上下文的访问量。现有的基于低秩和稀疏近似的次二次方法需要与密集注意力层相结合才能匹配 Transformer，这表现了能力上的差距。

本文提出了Hyena，一个次二次的注意力替代模块，通过交错地使用隐式参数化的**长卷积**和**数据控制门控**构建。在处理数千到数十万个标记的回归和推理任务中，Hyena在准确性上比依赖于状态空间和其他隐式和显式方法的操作符提高了超过50个百分点，与基于注意力的模型相当。

在语言建模标准数据集（WIKITEXT103和THE PILE）上为不需要密集注意力的架构设定了新的最优结果，以序列长度 2K为例，训练计算要求减少了20%。在序列长度为8K 和 64K 时，hynea运算速度是高度优化的注意力模型的两倍和 100 倍。

---

## 1.引言
大型Transformer已经在语言建模、视觉、音频、生物学和许多其他领域取得了一系列突破性进展。他的成功很大程度上依赖于缩放特性和上下文学习的出现，使其能够以上下文作为输入来推广到未见过的数据和任务。

> 缩放特性：要实现最佳计算训练，模型大小和训练词库数量应等比例缩放：模型大小每增加一倍，训练词库数量也应增加一倍。**Training Compute-Optimal Large Language Models，DeepMind**
> 上下文学习：大型语言模型表现出一定的执行上下文学习的能力，但目前尚不清楚成功的任务与训练数据中存在的内容之间的关系是什么。标准Transformer可以能够从上下文示例中学习看不见的线性函数，其性能可与最优最小二乘估计器相媲美。即使在两种形式的分布偏移下，上下文学习也是可能的：（i）在模型的训练数据和推理时提示之间，以及（ii）在推理过程中的上下文示例和查询输入之间。训练 Transformer 可以在上下文中学习更复杂的函数类——即稀疏线性函数、两层神经网络和决策树——其性能与特定于任务的学习算法相匹配或超过。**What can transformers learn in-context? a case study of simple function classes**

但它也有局限性，最值得注意的问题之一是计算成本，随着输入序列长度的增加而迅速增长。该成本与序列长度L的平方成比例增加，这对于模型可以考虑的上下文量设定了严格限制。
![[hyena.assets/Pasted image 20240618180100.png|500]]

突破二次限制是迈向深度学习的新可能性的关键步骤。

减少模型中注意力计算成本的努力主要涉及线性化、低秩和稀疏逼近的使用。这些方法在表达能力和速度之间引入了一种折衷，需要与标准的注意力层相结合才能达到Transformer的质量.

越来越多的证据表明，注意力机制在语言处理中只利用了其二次能力的一小部分。

>1. **Hungry Hungry Hippos: Towards Language Modeling with State Space Models**
>H3模型通过将两层SSM堆叠，结合输入和输出的乘法项，在很多情况下表现得非常接近甚至优于Transformer。同时，H3模型由于其线性复杂度，相比于Transformer在长序列处理上具有显著的计算优势。通过引入FlashConv算法以及针对长序列的优化，SSM能够实现与注意力机制相媲美的性能，而且计算效率更高。
>2. **In-context learning and induction heads**
>![[hyena.assets/Pasted image 20240618201011.png|675]]
>论据 1（宏观共同认知）： transformer语言模型在训练初期会经历一个 "阶段性变化"，在此期间会形成induction heads，同时上下文学习分数也会显著提高。
>论据 2（宏观共扰）： 当改变 transformer的结构，从而改变induction heads能否形成（以及何时形成）时，语境中学习的显著提高也会发生精确匹配的变化。
>论据 3（直接消融）：  当我们在测试时直接 "消除 "小模型中的induction heads时，上下文中学习性能会大大减少。
>论据 4（induction heads通用性的具体例子）： 尽管从复制文字序列的角度对induction heads进行了非常狭义的定义，但根据经验观察到，这些induction heads似乎还能实现更复杂类型的情境内学习，包括高度抽象的行为，因此它们可以解释大部分的情境内学习。
>论据5（induction head通用性的机制合理性）： 对于小型模型，可以从机制上解释induction heads是如何工作的，并能证明它们有助于情境中学习。此外，实际的运作机制表明，我们可以通过自然的方式重新利用它来进行更普遍的情境学习。
>论证 6（从小模型到大模型的连续性）： 在前5个论点中，induction heads解释情境学习的理由在小模型中比在大模型中更充分。然而，许多与induction heads和情境学习相关的行为和数据从小型模型到大型模型都是平滑连续的，这表明最简单的解释是机制是相同的。

提出问题：在大规模情况下，是否存在能够与注意力质量相匹配的次二次型运算符？

基于高效的次二次性原语的组合，如逐元素乘法（门控）和长卷积（即滤波器尺寸与输入一样长）获得了一个积极的答案。依靠一系列基于最新的关于“机械性可解释性”的研究工作的“目标推理任务”，如记忆和归纳，提炼出与性能相关的注意力的三个属性以及与现有次二次方法的质量差距：

**a.数据控制：** 注意力机制实现了具有表现力的基于数据控制的线性算子，可在单个程序块中对整个线性函数系列进行编码。Dissecting Neural ODEs

**b.次线性参数缩放：** 注意力层的参数数量与序列长度脱钩，允许 Transformers 在注意力层之间的前馈神经网络（FFN）等其他地方分配更多参数。

**c. 不受限制的上下文：** 对于给定的输入，注意力具有不受限制的上下文，即它可以近似任何两个输入之间的依赖关系，而不受任意限制，如局部性（使用掩码的情况除外，如自回归模型）。

**Hyena 层级结构**  由两个高效的次二次原语组成：**长卷积**和**逐元素乘法门**
![[hyena.assets/Pasted image 20240619140205.png|675]]
![[hyena.assets/Pasted image 20240619140323.png]]
1.Hyena运算符的等效定义是一个数据控制矩阵的分解，即矩阵的条目是输入的函数。
2.通过利用快速卷积算法高效地评估Hyena运算符，而无需形成完整的矩阵。
实验结果表明，Hyena运算符能够显著缩小与注意力在规模上的质量差距，以较小的计算预算实现类似的困惑度和下游性能，且不使用注意力的混合策略。

**能力差距**：语言建模性能相关的推理任务，扩充了基本的机械解释基准测试，其他在任务复杂度增加时（例如词汇表大小增加时）模型性能迅速下降的额外任务，得到最佳参数化方式，准确性相比其他运算符提高超过50%。

**语言和视觉规模化任务** :
十亿参数级别上对Hyena进行了自回归语言建模的测试，为传统数据集（WIKITEXT103和THE PILE）中无密集注意力体系结构设定了新的最佳性能,并与transformer质量相匹配。
335M参数级，THE PILE，与transformer困惑度匹配，并将浮点运算数（FLOPS）的总数减少了20%。
> 一个 825 GiB 的英语文本语料库，旨在训练大规模语言模型。该桩由22个不同的高质量子集构成 - 包括现有的和新建的 - 其中许多来自学术或专业来源。

用Hyena替换VIT中的注意力，在图像分类任务中，当从头开始在imagenet-1k上进行训练时，Hyena能够与注意力在准确率上相匹配。

**更长的上下文** 长度为8192的情况下，与密集自注意力相比，速度提升了5倍，比高度优化的FlashAttention提高了2倍，并在序列长度为64K的情况下，比FlashAttention快了100倍，避免标准注意力内存问题。

## 2.预备和相关知识

离散卷积是一个具有两个参数的函数：长度为L的输入信号u和可学习的滤波器h。
用一个（可能是无限长的）可测滤波器h与长度为L的输入信号u进行线性（非周期性）卷积，定义为
$$
        y_t = (h * u)_t = \sum_{n=0}^{L-1} h_{t -n} u_n.
$$
一般而言，$u_t\in\mathbb{R}^D$，其中$D$是信号的宽度，即$\textit{通道}$的数量。这里分析专门针对$\textit{单输入单输出}$（SISO）层，即$D=1$的情况。$\textit{多输入多输出}$（MIMO）情况可以直接推导得到。
所以输入信号可以表示为一个向量$u\in\mathbb{R}^L$，卷积操作可以看作输入向量和由滤波器$h$引起的Toeplitz卷积核矩阵$\newcommand{\sS}{\mathsf{S}}\sS_h \in \mathbb{R}^{L \times L}$之间的矩阵向量乘积。
$$
    \begin{aligned}
        (h * u) =
        \begin{bmatrix}
            h_0 & h_{-1} & \cdots & h_{-L+1} \\
            h_1 & h_0 & \cdots & h_{-L+2} \\
            \vdots & \vdots & \ddots & \vdots \\
            h_{L-1} & h_{L-2} & \cdots & h_{0}
        \end{bmatrix}
        \begin{bmatrix}
            u_0\\
            u_1\\
            \vdots\\
            u_{L-1}
        \end{bmatrix}
    \end{aligned}
$$
### 2.1 显式卷积和隐式卷积
参数化和优化卷积滤波器$h_t$是深度学习和更广泛的信号处理中的标准过程。

CNNs的经典方法是直接优化滤波器响应函数$h_t$在$M$个预定步骤中的值，我们称之为$\textit{显式参数化}$。
$M$被称为$\textit{滤波器大小}$，通常远小于输入序列的长度$M \ll L$。这样的滤波器在信号处理中被称为$\textit{有限脉冲响应}$（FIR）滤波器。

FIR滤波器是局部的，并且可以捕捉最多相隔$M$步的输入之间的依赖关系。它们的主要优势是速度快，复杂度为$\mathcal{O}(ML)$。FIR滤波器的参数数量与滤波器大小呈线性关系，这可能会带来计算上的限制。

为了将参数数量与滤波器大小分离开，将滤波器$h_t$表示为时间步$t$的参数函数，即$h_t = \gamma_\theta(t)$，其中$\theta$是函数$\gamma_\theta$的参数。这种参数化被称为$\textit{隐式参数化}$。

隐式参数化的一种选择是将$h$选择为线性状态空间模型（SSM）的响应函数，由一阶差分方程描述：
$$
\newcommand{\sA}{\mathsf{A}}
\newcommand{\sB}{\mathsf{B}}
\newcommand{\sC}{\mathsf{C}}
\newcommand{\sD}{\mathsf{D}}
\newcommand{\sM}{\mathsf{M}}
\newcommand{\sW}{\mathsf{W}}
\newcommand{\R}{\mathbb{R}}

\begin{equation*}
    \begin{aligned}
        x_{t+1} &= \sA x_t + \sB u_t &&~~ \text{state equation} & \\
        y_t &= \sC x_t + \sD u_t &&~~ \text{output equation} &
    \end{aligned}
\end{equation*}
$$
这里，对于$x_0 = 0$的方便选择使得输入输出映射成为一个简单的卷积。
$$
\begin{aligned}
        y_t & =\sum_{n=0}^{t}\left(\sC\sA^{t - n}\sB + \sD \delta_{t-n}\right)u_n
\end{aligned}
$$
其中，$\delta_t$表示Kronecker delta。
![[hyena.assets/Pasted image 20240619160831.png|700]]
于是滤波器$h$识别为
$$
 t\mapsto h_t =
    \begin{cases}
        0 & t<0\\
        \sC \sA^t \sB + \sD\delta_t & t\geq 0
    \end{cases}
$$
其中$\sA，\sB，\sC$和$\sD$的条目是滤波器的学习参数。在网络设计方面，SSM的自由度是状态的维度和矩阵的结构。

SSM是一种典型的例子，说明了具有次线性参数数量的长卷积如何改善长序列的深度学习模型。

其他隐式方法包括将滤波器的参数化表示为从$t$（位置编码）到滤波器响应的映射，即$\gamma_\theta：t \mapsto h_t=\gamma_\theta(t)$，例如使用前馈神经网络。

> [!Long convolutions and memory:]
>  单个计算单元的“记忆”的粗略代理量是其能访问多远的过去信息以生成某一步的输出。这可以通过一定步骤中非零项的数量$\partial y_t / \partial u_{t-n}$（其中$n = 0, \ldots, t$）来粗略量化。对于CNN来说，滤波器的记忆等同于滤波器的大小$M$，因为$\partial y_t / \partial u_{t-n} = h_{n}$。因此，CNN的总记忆容量与模型参数的数量成比例。另一方面，隐式参数化允许我们将每个滤波器的记忆从参数数量中分离出来，并由学习到的参数隐式地控制滤波器的长度。在状态空间模型（SSM）中，$\partial y_t / \partial u_{t-n} = \sC \sA^n \sB$，记忆范围完全由$\sA$的谱半径决定，并且可以通过训练过程精细调整。另一方面，参数的数量则控制记忆单元的“表现力”，例如组成$h_t$的基函数的数量。
> 

$\textbf{快速卷积算法}$: 
$$
        y_t = (h * u)_t = \sum_{n=0}^{L-1} h_{t -n} u_n.
$$
简略看，传统卷积的渐进时间复杂度为$O(L^2)$。
库利-图基（Cooley-Tukey）快速傅里叶变换（FFT）算法最早的应用之一就是比传统方法计算公式更快地实现卷积。
实现在次二次时间内进行$\textit{快速长卷积}$的常见方法是使用FFT算法。该方法首先通过对输入和滤波器序列进行适当的零填充，将$\textit{非周期性}$卷积转化为$\textit{循环}$卷积。得到的核$\hat\sS_h$是一个循环矩阵，并通过离散傅里叶基进行对角化。
$$
\hat \sS_h = \sW^{-1} \sD_{H} \sW
$$
其中，$\sW$是DFT矩阵，$\sW_{tt'} = z^{-t}$，$z = e^{i2\pi t' / L}$，$H$是零填充滤波器$h$的DFT，$H = \sW \text{pad}(h)$。
卷积计算被执行为
$$
\begin{equation*}
    \begin{aligned}
        {\sf pad}(y) &= \hat \sS_h {\sf pad}(u) \\
        &= \sW^{-1}\sD_H \sW ~{\sf pad}(u)\\
        &= {\sf iFFT}(\sD_H {\sf FFT}({\sf pad}(u)))
    \end{aligned}
\end{equation*}
$$
其中，$\sD_H$ 是以 $\sW h$ 为对角线的矩阵。上述被称为离散傅里叶变换（DFT）的卷积定理。在 ${\sf FFTConv}$形式中，卷积可以在不实例化算子 $\hat \sS_h$ 的情况下进行，其渐进复杂度与快速傅里叶变换（FFT）相同， $O(L\log_2 L)$。  
![[hyena.assets/Pasted image 20240620194702.png|450]]
### 2.2 自注意力算子
在Transformer模型的核心是$\textit{多头注意力}$（multi-head attention，MHA）机制。
对于长度为$L$的序列$u\in\R^{L\times D}$，$\textit{缩放自注意力}$（scaled self-attention）的每个$\textit{头}$是一个从$\R^{L\times D}$到$\R^{L\times D}$的映射，执行以下操作：
$$
\begin{equation}
    \begin{aligned}
      \sA(u) &= {\sf SoftMax}\left(\tfrac{1}{\sqrt{D}}u \sM_q \sM^\top_k u^\top \right)\\
      y &= {\sf SelfAttention}(u) \\
        &= \sA(u)u \sM_v,
    \end{aligned}
\end{equation}
$$
其中，$\newcommand{\x}{\times}\sM_q, \sM_k, \sM_v\in\R^{D\x D}$ 是可学习的线性投影矩阵，${\sf SoftMax}$ 按行应用。
注意力机制参数化了一个$\textbf{密集线性算子的系列}$
对于输入 $u$，通过 $u$ 的投影 $\sA(u)$ 进行索引。
将这种类型的算子称为$\textit{数据控制}$ 型，因为它们通过 $u$ 非线性地定义了一个线性变换 $u \mapsto y$。这种方法在 $u$ 中产生了具有表达力的非线性算子，并且假设，与其他机制(induction heads)一起，这有助于通过利用上下文从而学习到$\textit{上下文中的知识}$，从而适应从未见过的任务。
即深度学习中的：$\textit{查询}$  $q=u\sM_q$，$\textit{关键字}$  $k=u\sM_k$，$\textit{数值}$  $v = u\sM_v$。
常常将注意力算子重写为 $y = \sA(q,k)v$。

**Remark** 2.1
类似于隐式卷积，自注意力机制在不增加参数数量的前提下能够访问远距离的信息：它以 $\newcommand{\cO}{\mathcal{O}} \cO(L^2)$ 的计算复杂度处理整个序列。

### 2.3 次二次运算子

注意力机制替代方案
现有的方法通过改变数据控制的实现方式总结，即通过$u$非线性定义运算子，然后应用于$v$。
· Attention-Free Transformers (AFTs)
消除了点积自注意力,在AFT层中，键（key）和值（value）首先与一组学习到的位置偏差相结合，然后通过逐元素乘法与查询（query）相乘。这种新的操作在内存复杂度上与上下文大小和特征维度呈线性关系，因此适用于大型输入和模型规模。
![[hyena.assets/Pasted image 20240622133312.png|500]]
![[hyena.assets/Pasted image 20240622133236.png|450]]
