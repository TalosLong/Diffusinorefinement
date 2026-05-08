# 基于条件扩散模型的医学图像分割精细化方法：扩散模块设计与实现


**摘　要**

本章详细阐述基于条件扩散模型的医学图像分割精细化系统中扩散模块的各核心组件。该系统采用两阶段架构：第一阶段由粗分割网络（CPUNet）产生初始分割概率图；第二阶段引入条件去噪扩散概率模型对分割边界进行精细化优化。扩散模块涵盖高斯扩散调度器、时间步嵌入编码器、条件残差块、自注意力模块、先验特征调制模块（PFMM）、条件特征融合模块（CFFM）、空间与通道注意力模块（SACM）、多尺度图像特征编码器以及小波空间变换器（WS-Former）等关键组件。本章按模块逐一阐述各组件的设计原理、数学推导及实现细节。

**关键词**：条件扩散模型；医学图像分割；去噪扩散概率模型；小波变换；注意力机制


## 第一节　引言

医学图像分割是计算机辅助诊断与治疗规划的核心基础任务。随着深度学习技术的迅速发展，以U-Net及其变体为代表的卷积神经网络模型在医学图像分割领域取得了显著成果。然而，此类模型所产生的粗分割结果在病变组织边界区域往往存在轮廓不精确、细节缺失等问题，制约了临床应用的进一步推广。

为解决上述问题，本系统在粗分割网络的基础上，引入去噪扩散概率模型（Denoising Diffusion Probabilistic Models，DDPM）作为后处理精细化手段。扩散模型通过模拟数据的逐步加噪与去噪过程，具备强大的概率生成能力，能够在保留粗分割整体结构的同时，逐步消除边界噪声，恢复精细的解剖轮廓。系统整体流程如下：

$$
\text{输入图像} \xrightarrow{\text{预处理}} \text{CPUNet（粗分割）} \xrightarrow{\text{扩散精细化}} \text{后处理} \xrightarrow{} \text{最终分割结果}
$$

扩散优化模块是本系统的核心创新所在，其内部由多个精心设计的功能子模块有机组合而成。本章以下各节将依次对各组件的设计原理与数学公式进行系统性阐述。


## 第二节　高斯扩散调度器

### 2.1　概述

高斯扩散调度器是扩散模型的数学基础，负责统一管理前向扩散（加噪）过程与反向扩散（去噪）过程中的噪声水平调度，并为网络训练和推理采样提供必要的统计量支撑。

### 2.2　前向扩散过程

前向扩散过程定义了如何向干净数据 $x_0$（真实分割掩码）中逐步添加高斯噪声，使其演变为趋近于标准正态分布的随机变量。其条件分布定义为：

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)\, \mathbf{I}\right) \tag{2-1}$$

其中，$\beta_t$ 为噪声调度参数，按预设方案随时间步单调递增；$\alpha_t = 1 - \beta_t$；$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ 为累积乘积。利用重参数化技巧，可在任意时间步 $t$ 直接对 $x_t$ 进行闭式采样：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) \tag{2-2}$$

### 2.3　噪声调度方案

本系统支持两种噪声调度方案以适应不同训练场景。

**（1）线性调度方案**

$$\beta_t = \beta_{\text{start}} + \frac{t}{T}\left(\beta_{\text{end}} - \beta_{\text{start}}\right) \tag{2-3}$$

默认超参数设置为 $\beta_{\text{start}} = 10^{-4}$，$\beta_{\text{end}} = 0.02$，总步数 $T = 1000$。

**（2）余弦调度方案**

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right) \tag{2-4}$$

其中 $s = 0.008$ 为防止 $\beta_t$ 在 $t$ 趋近于零时过小而引入的偏移常数。余弦调度在扩散过程两端变化较为平缓，有助于模型在低噪声水平下更充分地学习细节。

### 2.4　反向过程后验参数

反向过程的真实后验分布 $q(x_{t-1} \mid x_t, x_0)$ 为高斯分布，其均值与方差可解析计算：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t \tag{2-5}$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\, \beta_t \tag{2-6}$$

上述参数在模型初始化阶段完成预计算并缓存，以提升训练和推理效率。

### 2.5　DDIM 加速采样

为降低推理阶段的计算开销，本系统采用去噪扩散隐式模型（Denoising Diffusion Implicit Models，DDIM）<sup>[2]</sup> 作为推理采样策略。DDIM 在不改变边缘分布的前提下，允许在远小于 $T$ 的子时间步序列上完成确定性或随机采样。给定网络预测的 $\hat{x}_0$ 及推导得到的噪声估计 $\hat{\epsilon}$，单步去噪更新规则为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \hat{\epsilon} + \sigma_t \cdot z \tag{2-7}$$

其中，$\sigma_t = \eta \sqrt{\dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \dfrac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$，$z \sim \mathcal{N}(0, \mathbf{I})$。当 $\eta = 0$ 时，采样过程退化为完全确定性的，消除了随机性带来的方差。本系统默认以 20～50 步 DDIM 采样替代完整的 1000 步 DDPM 采样，在推理效率与分割精度之间取得良好平衡。


## 第三节　时间步嵌入编码器

### 3.1　概述

时间步嵌入编码器负责将离散的扩散时间步 $t \in \{0, 1, \ldots, T-1\}$ 映射为连续的高维向量表示，使去噪网络能够感知当前的噪声水平，并据此调整特征提取策略。

### 3.2　正弦位置编码

参考 Transformer 中的经典位置编码设计<sup>[6]</sup>，时间步嵌入采用正弦函数族对不同频率进行编码：

$$\text{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right) \tag{3-1}$$

$$\text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right) \tag{3-2}$$

其中 $d$ 为嵌入维度（默认取 256），$i$ 为维度索引。该编码方案能够为每个时间步赋予唯一且平滑变化的频率指纹，有利于网络学习跨时间步的连续性。

正弦编码随后经过一个两层多层感知机（MLP）进行非线性变换：

$$e_t = W_2 \cdot \text{SiLU}(W_1 \cdot \text{PE}(t) + b_1) + b_2 \tag{3-3}$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$ 为 Sigmoid 线性单元激活函数。变换后的嵌入向量 $e_t$ 维度与去噪网络各残差块内部通道数保持一致，以实现逐层无缝注入。


## 第四节　条件残差块

### 4.1　概述

条件残差块是条件去噪网络的基本构建单元。在标准残差连接的基础上，该模块引入了时间步条件注入机制，使网络能够根据当前扩散时间步动态调整特征响应，从而在不同噪声水平下展现差异化的去噪行为。

### 4.2　结构设计

设输入特征为 $x$，时间步嵌入为 $e_t$，残差块的完整计算流程如下：

$$h = \text{Conv}_{3 \times 3}\!\left(\text{SiLU}\!\left(\text{GN}(x)\right)\right) \tag{4-1}$$

$$h \leftarrow h + W_t \cdot \text{SiLU}(e_t) \tag{4-2}$$

$$h = \text{Conv}_{3 \times 3}\!\left(\text{Dropout}\!\left(\text{SiLU}\!\left(\text{GN}(h)\right)\right)\right) \tag{4-3}$$

$$\text{output} = h + \text{Shortcut}(x) \tag{4-4}$$

其中，$\text{GN}(\cdot)$ 为分组归一化（Group Normalization），分组数取 8；时间步嵌入通过线性投影 $W_t \in \mathbb{R}^{d_t \times C_{\text{out}}}$ 扩展到与特征图通道数相同的维度后，以广播加法的形式作用于空间维度上的所有位置，即

$$h' = h + (W_t \cdot \text{SiLU}(e_t))_{:,\,:,\, \text{None},\, \text{None}} \tag{4-5}$$

当输入与输出通道数不一致时，捷径连接 $\text{Shortcut}(\cdot)$ 采用 $1 \times 1$ 卷积进行通道维度匹配；否则使用恒等映射。


## 第五节　自注意力模块

### 5.1　概述

自注意力模块用于在特征图内部建模长距离空间依赖关系，以弥补局部卷积操作感受野有限的不足。本系统在条件去噪网络的瓶颈层及特定低分辨率层中嵌入该模块。

### 5.2　多头自注意力计算

设输入特征 $X \in \mathbb{R}^{B \times C \times H \times W}$，自注意力模块的计算流程如下。

首先对输入进行分组归一化以稳定训练：

$$\hat{X} = \text{GN}(X) \tag{5-1}$$

随后将空间维度展平，转换为序列形式 $\hat{X}_{\text{flat}} \in \mathbb{R}^{B \times HW \times C}$，并以此计算多头注意力（默认注意力头数为 4）：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V \tag{5-2}$$

最后通过残差连接与输入相加，得到最终输出：

$$\text{output} = X + \text{Proj}\!\left(\text{Attention}(\hat{X}_{\text{flat}})\right) \tag{5-3}$$

考虑到计算复杂度与空间尺寸成平方关系，本系统仅在特征图分辨率为 $16 \times 16$ 和 $8 \times 8$ 的层级应用自注意力，以在计算效率与全局表征能力之间取得合理权衡。


## 第六节　先验特征调制模块

### 6.1　概述

先验特征调制模块（Prior Feature Modulation Module，PFMM）是条件去噪网络的输入端调制单元。该模块将粗分割网络（CPUNet）生成的 $K$ 类概率图 $P$ 作为先验空间信息，对初始卷积后的特征 $Z$ 进行类别感知的加权调制，使去噪网络在处理初始阶段即能感知粗分割结果的空间分布规律，从而将后续去噪过程的注意力引导至需要精细化的边界区域。

### 6.2　数学推导

设 $P \in \mathbb{R}^{B \times K \times H \times W}$ 为 $K$ 类概率图，$Z \in \mathbb{R}^{B \times C \times H \times W}$ 为待调制的特征张量。PFMM 的计算过程由以下四步构成。

**（1）概率图通道拆分**

$$P = [P_1,\, P_2,\, \ldots,\, P_K], \quad P_k \in \mathbb{R}^{B \times 1 \times H \times W} \tag{6-1}$$

**（2）类别加权特征拼接**

$$Z' = \text{Concat}(P_1 \odot Z,\; P_2 \odot Z,\; \ldots,\; P_K \odot Z) \in \mathbb{R}^{B \times KC \times H \times W} \tag{6-2}$$

其中 $\odot$ 表示逐元素乘法，利用广播机制在通道维度实现类别感知的特征加权。

**（3）深度可分离卷积与残差归一化**

$$Z'' = \text{LN}\!\left(\text{DWConv}(Z') \oplus Z\right) \tag{6-3}$$

其中 $\text{DWConv}(\cdot)$ 为深度可分离卷积（逐通道卷积后接 $1 \times 1$ 逐点卷积，将通道数由 $KC$ 映射回 $C$），$\oplus$ 表示残差加法，$\text{LN}(\cdot)$ 为层归一化。

**（4）最终输出**

$$Z''' = \text{ReLU}\!\left(\text{LN}\!\left(\text{Conv}_{3 \times 3}(Z'')\right)\right) \tag{6-4}$$

### 6.3　设计分析

PFMM 的核心设计思路在于：通过将各类别概率图分别与特征图相乘，高概率区域（即粗分割置信度高的区域）对应的特征得到增强，低概率区域的特征则被相对抑制。这一机制使去噪网络在初始特征提取阶段即融入了粗分割的空间先验，有效缩减了后续去噪需要修正的搜索空间。


## 第七节　条件特征融合模块

### 7.1　概述

条件特征融合模块（Conditional Feature Fusion Module，CFFM）负责将多尺度图像特征编码器所提取的低级图像特征注入到条件去噪网络编码器的对应层级中。该模块作用于编码器的前三个尺度层（对应较高分辨率的层级），确保去噪网络在特征编码阶段能够持续参考原始图像的纹理细节与边缘梯度信息，从而在分割边界处实现更高精度的精细化。

### 7.2　数学推导

设 $F \in \mathbb{R}^{B \times C_f \times H \times W}$ 为条件去噪网络编码器某层的特征图，$X_{\text{low}} \in \mathbb{R}^{B \times C_x \times H \times W}$ 为图像特征编码器对应尺度的低级特征（已经过 $1 \times 1$ 卷积投影与空间插值，确保尺度与通道数一致）。CFFM 的计算过程如下。

**（1）特征拼接**

$$F' = \text{Concat}(F,\; X_{\text{low}}) \in \mathbb{R}^{B \times (C_f + C_x) \times H \times W} \tag{7-1}$$

**（2）深度可分离卷积提取融合特征**

$$F'' = \text{ReLU}\!\left(\text{LN}\!\left(\text{DWConv}(F')\right)\right) \tag{7-2}$$

其中 $\text{DWConv}(\cdot)$ 将拼接后的通道数由 $C_f + C_x$ 映射回 $C_f$。

**（3）残差连接与归一化**

$$F''' = \text{LN}(F'' \oplus F) \tag{7-3}$$

### 7.3　设计分析

低级图像特征（如边缘响应、局部纹理）是指导分割边界精确定位的重要依据，而扩散去噪过程中的深层语义特征本身对这些局部细节缺乏直接感知能力。CFFM 通过深度可分离卷积对融合特征进行高效非线性变换，同时利用残差连接保留编码器原有的语义表达，实现了局部细节与去噪特征的有机融合。


## 第八节　空间与通道注意力模块

### 8.1　概述

空间与通道注意力模块（Spatial and Channel Attention Module，SACM）作用于条件去噪网络编码器的最深层（即最低分辨率层级），负责将图像特征编码器提取的高级语义特征以注意力驱动的方式融入编码器的深层表示。与前述 CFFM 采用拼接与卷积的显式融合策略不同，SACM 通过双路注意力机制（通道注意力与空间注意力）实现更具选择性的自适应特征整合。

### 8.2　通道注意力分支

设 $X_{\text{high}} \in \mathbb{R}^{B \times C_x \times H \times W}$ 为图像特征编码器的高级特征。首先经卷积投影统一通道维度：

$$X' = \text{Conv}_{3 \times 3}(X_{\text{high}}) \in \mathbb{R}^{B \times C_s \times H \times W} \tag{8-1}$$

对 $X'$ 施以全局最大池化与全局平均池化，分别得到：

$$X'_{\text{max}} = \text{GAP}_{\max}(X') \in \mathbb{R}^{B \times C_s}, \quad X'_{\text{avg}} = \text{GAP}_{\text{avg}}(X') \in \mathbb{R}^{B \times C_s} \tag{8-2}$$

将两路池化结果分别送入共享参数的两层 MLP（通道压缩比 $r = 16$）并求和，经 Sigmoid 激活后得到通道注意力权重：

$$\text{CA} = \sigma\!\left(\text{MLP}(X'_{\text{max}}) + \text{MLP}(X'_{\text{avg}})\right) \in \mathbb{R}^{B \times C_s} \tag{8-3}$$

其中，$\text{MLP}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$，$W_1 \in \mathbb{R}^{C_s \times (C_s/r)}$，$W_2 \in \mathbb{R}^{(C_s/r) \times C_s}$。

### 8.3　空间注意力分支

对 $X'$ 在通道维度上分别取最大值与均值，得到空间统计特征：

$$X'_{\max,\text{sp}} = \max_{c}(X') \in \mathbb{R}^{B \times 1 \times H \times W}, \quad X'_{\text{avg,sp}} = \text{mean}_{c}(X') \in \mathbb{R}^{B \times 1 \times H \times W} \tag{8-4}$$

将上述两者在通道维度扩展后拼接，经深度可分离卷积与 Sigmoid 激活得到空间注意力图：

$$X_{\text{cat}} = \text{Concat}(X'_{\max,\text{sp}},\; X'_{\text{avg,sp}}) \in \mathbb{R}^{B \times 2C_s \times H \times W} \tag{8-5}$$

$$\text{SA} = \sigma\!\left(\text{DWConv}_{7 \times 7}(X_{\text{cat}})\right) \in \mathbb{R}^{B \times 1 \times H \times W} \tag{8-6}$$

### 8.4　双路注意力特征融合

设 $S \in \mathbb{R}^{B \times C_s \times H \times W}$ 为条件去噪网络编码器深层特征，最终融合输出为：

$$S' = \text{Conv}_{3 \times 3}(S) \tag{8-7}$$

$$S'' = \text{LN}\!\left(S \oplus (\text{CA} \odot S') \oplus (\text{SA} \odot S')\right) \tag{8-8}$$

### 8.5　设计分析

SACM 的设计思路借鉴了卷积块注意力模块（Convolutional Block Attention Module，CBAM）<sup>[5]</sup> 的框架思想，但将其由单模态特征自增强改造为跨模态特征融合机制。通道注意力通过感知高级图像语义特征的通道重要性分布，选择性地强化去噪特征的语义相关通道；空间注意力则通过识别图像高级特征中具有判别性的空间区域，引导去噪网络聚焦于解剖结构的关键位置。两路注意力协同作用，有效弥补了深层去噪特征在语义感知层面的不足。


## 第九节　多尺度图像特征编码器

### 9.1　概述

多尺度图像特征编码器基于 ResNet-V2 预激活残差网络架构，从原始输入图像中提取四个分辨率层级的特征图，分别为后续各条件注入模块（CFFM 和 SACM）提供不同粒度的图像先验信息。

### 9.2　网络结构

编码器由一个根节点卷积层（Root）与三个残差阶段（Stage）组成，各阶段采用预激活瓶颈残差块（Pre-Activation Bottleneck Block）堆叠而成。各阶段的输出规格如表9-1所示。

**表9-1　多尺度图像特征编码器各阶段输出规格**

| 阶段 | 输出通道数 | 输出空间分辨率 | 残差块数量 | 对应条件注入模块 |
|:----:|:---------:|:-------------:|:---------:|:--------------:|
| Root | 64 | $H/2 \times W/2$ | — | CFFM（第0层级） |
| Stage 1 | 256 | $H/4 \times W/4$ | 3 | CFFM（第1层级） |
| Stage 2 | 512 | $H/8 \times W/8$ | 4 | CFFM（第2层级） |
| Stage 3 | 1024 | $H/16 \times W/16$ | 6 | SACM（第3层级） |

### 9.3　预激活瓶颈残差块

预激活瓶颈残差块在标准瓶颈结构的基础上，将批归一化（Batch Normalization）和激活函数置于卷积操作之前（即"先归一化再激活"），有助于改善深层网络的梯度传播特性<sup>[7]</sup>。设输入为 $x$，计算流程如下：

$$y = \text{ReLU}\!\left(\text{GN}_1\!\left(\text{Conv}_{1 \times 1}(x)\right)\right) \tag{9-1}$$

$$y = \text{ReLU}\!\left(\text{GN}_2\!\left(\text{Conv}_{3 \times 3}(y)\right)\right) \tag{9-2}$$

$$y = \text{GN}_3\!\left(\text{Conv}_{1 \times 1}(y)\right) \tag{9-3}$$

$$\text{output} = \text{ReLU}\!\left(y + \text{Downsample}(x)\right) \tag{9-4}$$

此处所有卷积层均采用权重标准化卷积（Weight Standardization），对卷积核权重实施标准化处理：

$$\hat{w} = \frac{w - \mu_w}{\sqrt{\sigma_w^2 + \epsilon}} \tag{9-5}$$

权重标准化与分组归一化的结合能够进一步提升特征表示的稳定性，在小批量训练场景下效果尤为显著。

### 9.4　通道对齐投影层

由于图像特征编码器各阶段的输出通道数（64、256、512、1024）与条件去噪网络编码器各层级通道数（64、128、256、512）存在不匹配，系统在两者之间引入 $1 \times 1$ 卷积投影层进行通道对齐：

$$\hat{F}_i = \text{ReLU}\!\left(\text{GN}\!\left(\text{Conv}_{1 \times 1}(F_i)\right)\right) \tag{9-6}$$

当两者通道数相同时，投影层退化为恒等映射以避免不必要的参数开销。此外，若编码器特征与去噪网络特征的空间尺寸不一致，系统还会通过双线性插值进行空间对齐，以保证各条件注入模块能够正常运作。


## 第十节　小波空间变换器

### 10.1　概述

小波空间变换器（Wavelet-Space Transformer，WS-Former）是本系统扩散精细化模块的核心创新组件，与 SACM 协同作用于条件去噪网络编码器的最深层级。该模块利用一维离散小波变换（Discrete Wavelet Transform，DWT）将特征图分解至频率域，在小波子带空间中分别对低频语义分量与高频细节分量执行交叉注意力交互，并通过时间步自适应门控机制动态调节各子带的信号权重，最终经逆小波变换（Inverse DWT，IDWT）重建精细化的空间域特征。

### 10.2　二维离散小波分解

对输入特征 $F \in \mathbb{R}^{B \times C \times H \times W}$ 实施一级 Haar 小波变换，得到四个子带分量：

$$\text{DWT}(F) = \left(F_{\text{LL}},\; \left[F_{\text{LH}},\; F_{\text{HL}},\; F_{\text{HH}}\right]\right) \tag{10-1}$$

其中，$F_{\text{LL}} \in \mathbb{R}^{B \times C \times H/2 \times W/2}$ 为低频近似子带，保留了特征图的主要语义结构；$F_{\text{LH}},\, F_{\text{HL}},\, F_{\text{HH}} \in \mathbb{R}^{B \times C \times H/2 \times W/2}$ 分别为水平、垂直与对角方向的高频细节子带，编码了边缘与纹理等局部细节信息<sup>[8]</sup>。

### 10.3　小波域交叉注意力

对扩散去噪特征 $N$ 与条件图像特征 $C$ 分别执行 DWT 分解后，在对应子带之间进行跨模态交叉注意力交互。交叉注意力模块的 Query 来自去噪特征子带（引导精细化方向），Key 与 Value 均来自条件特征子带（提供参考信息）。

**（1）低频子带语义对齐**

$$\hat{N}_{\text{LL}} = \text{CrossAttn}(N_{\text{LL}},\; C_{\text{LL}}) + N_{\text{LL}} \tag{10-2}$$

其中，交叉注意力的计算式为：

$$Q = W_Q \cdot N_{\text{LL}},\quad K = W_K \cdot C_{\text{LL}},\quad V = W_V \cdot C_{\text{LL}} \tag{10-3}$$

$$\text{CrossAttn}(Q, K, V) = W_O \cdot \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \tag{10-4}$$

**（2）高频子带细节对齐**

三个高频子带（$\text{LH}$、$\text{HL}$、$\text{HH}$）以 Batch 维度合并后共享同一注意力模块参数进行并行处理：

$$\left[\hat{N}_{\text{LH}},\; \hat{N}_{\text{HL}},\; \hat{N}_{\text{HH}}\right] = \text{CrossAttn}_{\text{High}}\!\left(\left[N_{\text{LH}}, N_{\text{HL}}, N_{\text{HH}}\right],\; \left[C_{\text{LH}}, C_{\text{HL}}, C_{\text{HH}}\right]\right) + \left[N_{\text{LH}}, N_{\text{HL}}, N_{\text{HH}}\right] \tag{10-5}$$

### 10.4　时间自适应子带门控

去噪过程中不同时间步所需的频率信息侧重存在显著差异：在早期去噪阶段（$t$ 较大，噪声水平高）应优先依赖低频子带的语义结构信息以恢复整体轮廓；在后期精细化阶段（$t$ 较小，噪声水平低）则需增大高频子带的贡献以恢复边缘细节。为此，时间自适应子带门控模块根据当前时间步嵌入 $e_t$ 学习四个子带的动态加权系数：

$$\left[g_{\text{LL}},\; g_{\text{LH}},\; g_{\text{HL}},\; g_{\text{HH}}\right] = \sigma\!\left(\text{MLP}(e_t)\right) \in [0, 1]^{C \times 4} \tag{10-6}$$

门控加权后的子带输出为：

$$\hat{N}'_{\text{LL}} = g_{\text{LL}} \odot \hat{N}_{\text{LL}} \tag{10-7}$$

$$\hat{N}'_{\text{LH}} = g_{\text{LH}} \odot \hat{N}_{\text{LH}},\quad \hat{N}'_{\text{HL}} = g_{\text{HL}} \odot \hat{N}_{\text{HL}},\quad \hat{N}'_{\text{HH}} = g_{\text{HH}} \odot \hat{N}_{\text{HH}} \tag{10-8}$$

### 10.5　逆小波重构与特征输出

经门控处理后的子带特征通过逆小波变换重建为空间域特征：

$$F_{\text{out}} = \text{IDWT}\!\left(\hat{N}'_{\text{LL}},\; \left[\hat{N}'_{\text{LH}},\; \hat{N}'_{\text{HL}},\; \hat{N}'_{\text{HH}}\right]\right) \tag{10-9}$$

最终通过由两个 $1 \times 1$ 卷积层、批归一化与 SiLU 激活函数组成的特征投影网络，输出维度与输入保持一致的精细化特征图。

### 10.6　设计分析

传统扩散模型在欧氏空间域直接对特征进行整合，难以同时兼顾全局语义一致性与局部边缘精度。WS-Former 通过引入小波频率分解，将特征交互转移至各向异性的频率子带空间，实现了语义信息与细节信息的解耦处理。低频交叉注意力负责语义层面的粗对齐，高频交叉注意力则专注于边缘和纹理层面的细节对齐；时间自适应门控进一步确保去噪过程中的频率关注点能够随噪声水平动态自适应地调整，从而在整个去噪轨迹上均保持高质量的条件注入效果。


## 第十一节　条件去噪网络

### 11.1　概述

条件去噪网络是扩散精细化模块的主体架构，采用经典的编码器—解码器对称结构（U-Net），并在各层级系统性地集成了上述所有条件注入模块，以实现对多源条件信息的充分利用。

### 11.2　整体架构

网络的整体前向传播流程如下：输入噪声掩码 $x_t$ 经初始 $3 \times 3$ 卷积层映射为基础通道维度后，立即由 PFMM 利用粗分割先验概率图 $P$ 进行调制，将空间先验信息植入初始特征。

编码器由四个层级组成，各层级依次包含若干条件残差块、可选的自注意力模块以及条件注入单元（前三层级使用 CFFM，第四层级使用 SACM 与 WS-Former 的组合）；相邻层级之间通过步长为 2 的卷积下采样实现空间分辨率减半。各层级的输出特征通过跳跃连接保存，供解码器的对应层级使用。

瓶颈层由"条件残差块—自注意力模块—条件残差块"的串行结构组成，负责在最低分辨率层级对全局特征进行充分整合。

解码器与编码器对称，各层级通过转置卷积实现上采样，并将解码器特征与跳跃连接来的编码器特征在通道维度拼接后经条件残差块进一步处理。最终输出层由分组归一化、SiLU 激活和 $3 \times 3$ 卷积组成，生成预测的分割 logits。

### 11.3　网络超参数

网络的主要可配置超参数如表11-1所示。

**表11-1　条件去噪网络主要超参数**

| 超参数名称 | 默认值 | 含　义 |
|:---------:|:------:|:------:|
| 基础通道数 | 64 | 第一层级的特征通道数 |
| 通道倍增系数 | (1, 2, 4, 8) | 各层级相对于基础通道数的倍数 |
| 残差块数量 | 2 | 编码器每层级的条件残差块数量 |
| 注意力分辨率 | (16, 8) | 启用自注意力的特征图尺寸（单位：像素） |
| 时间嵌入维度 | 256 | 时间步嵌入向量的维度 |
| Dropout 比率 | 0.1 | 随机失活比率 |
| 输入图像分辨率 | 256 | 默认输入尺寸（像素） |

编码器各层级通道数依次为 64→128→256→512，解码器与之对称。

### 11.4　条件注入策略总览

本系统在条件去噪网络各层级采用差异化的条件注入策略，以匹配不同分辨率层级的特征语义层次，具体如表11-2所示。

**表11-2　条件去噪网络各层级条件注入策略**

| 编码器层级 | 特征图分辨率 | 图像特征来源 | 注入模块 | 注入方式 |
|:---------:|:-----------:|:-----------:|:-------:|:-------:|
| 输入层 | $H \times W$ | 粗分割概率图 $P$ | PFMM | 类别加权特征调制 |
| 第0层级 | $H/2$ | 编码器 Root（64 通道）| CFFM | 拼接+深度可分离卷积 |
| 第1层级 | $H/4$ | Stage 1（256 通道）| CFFM | 拼接+深度可分离卷积 |
| 第2层级 | $H/8$ | Stage 2（512 通道）| CFFM | 拼接+深度可分离卷积 |
| 第3层级 | $H/16$ | Stage 3（1024 通道）| SACM + WS-Former | 双路注意力+小波交叉注意力 |
| 全部层级 | — | 时间步 $t$ | 条件残差块内 MLP | 通道级加性注入 |


## 第十二节　扩散精细化优化器

### 12.1　概述

扩散精细化优化器是整个系统最顶层的封装模块，负责整合冻结的 CPUNet 粗分割网络、高斯扩散调度器与条件去噪网络，并提供统一的端到端训练与推理接口。在训练阶段，CPUNet 的所有参数均被冻结，系统仅对条件去噪网络的参数进行优化，有效降低了训练阶段的显存需求与优化复杂度。

### 12.2　训练过程

训练阶段的完整计算流程如下。

**（1）获取粗分割掩码**

$$P = \text{softmax}\!\left(\text{CPUNet}(I)\right) \tag{12-1}$$

**（2）条件增强**

以 50% 的概率对粗分割概率图施加随机形态学操作（膨胀或腐蚀），模拟推理阶段粗分割误差的分布特性，提升精细化网络对不完美粗分割结果的鲁棒性。此后加入幅度为 0.05 的高斯噪声扰动：

$$P' = \text{clamp}\!\left(P + \mathcal{N}(0,\; 0.05^2),\; 0,\; 1\right) \tag{12-2}$$

**（3）真值标签归一化**

将 Ground Truth 掩码 $M \in \{0, 1\}^{H \times W}$ 缩放至扩散模型所要求的 $[-1, 1]$ 值域：

$$x_0 = 2M - 1 \tag{12-3}$$

**（4）前向扩散采样**

在时间步域均匀采样 $t \sim \mathcal{U}(0, T)$，对真值标签执行前向扩散：

$$(x_t,\; \epsilon) = q(x_0,\; t) \tag{12-4}$$

**（5）去噪预测**

将噪声掩码 $x_t$、时间步 $t$、原始图像 $I$ 及条件增强后的粗分割掩码 $P'$ 一并送入条件去噪网络，直接预测干净掩码的 logits：

$$\hat{x}_0 = f_\theta(x_t,\; t,\; I,\; P') \tag{12-5}$$

**（6）混合损失函数**

本系统采用二元交叉熵损失与 Dice 损失的等权加和作为训练目标：

$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}} \tag{12-6}$$

其中，

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left[M_i \log \hat{p}_i + (1 - M_i)\log(1 - \hat{p}_i)\right] \tag{12-7}$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\displaystyle\sum_{i=1}^{N} \hat{p}_i M_i + \varepsilon}{\displaystyle\sum_{i=1}^{N} \hat{p}_i + \sum_{i=1}^{N} M_i + \varepsilon} \tag{12-8}$$

$\hat{p}_i = \sigma(\hat{x}_{0,i})$ 为预测概率，$\varepsilon = 10^{-5}$ 为数值稳定系数。Dice 损失对前景目标区域具有更强的关注能力，与交叉熵损失互补，有助于缓解医学图像中常见的类别不均衡问题。

### 12.3　推理过程

推理阶段基于 DDIM 加速采样策略，执行流程如下。

首先由 CPUNet 生成粗分割掩码 $P$，并将其归一化至扩散值域：$P_{\text{scaled}} = 2P - 1$。

与从纯高斯噪声出发的标准扩散推理不同，本系统采用**中间时间步启动策略**：默认将粗分割掩码视为 $x_0$，并在中间时间步 $t_{\text{start}} = \lfloor 0.7T \rceil$ 处对其添加对应噪声水平的噪声，作为 DDIM 反向采样的初始状态：

$$x_{t_{\text{start}}} = \sqrt{\bar{\alpha}_{t_{\text{start}}}}\, P_{\text{scaled}} + \sqrt{1 - \bar{\alpha}_{t_{\text{start}}}}\, \varepsilon,\quad \varepsilon \sim \mathcal{N}(0,\mathbf{I}) \tag{12-9}$$

随后在 $\{t_{\text{start}}, \ldots, 0\}$ 的等间距子时间步序列上执行 20～50 步 DDIM 反向去噪，最终将输出缩放回 $[0,1]$ 的概率值域：

$$\hat{M} = \text{clamp}\!\left(\frac{x_0 + 1}{2},\; 0,\; 1\right) \tag{12-10}$$

### 12.4　设计分析

中间时间步启动策略是本系统与标准扩散生成模型的重要区别。该策略将粗分割结果作为精细化的起点而非从随机噪声出发，有效避免了去噪过程在整体结构层面的无效探索；同时，从 $0.7T$ 而非 $T$ 开始采样将有效推理步数减少约 30%，在不显著牺牲精细化质量的前提下大幅提升了推理效率。


## 第十三节　模型规模与推理效率对比分析

为定量评估引入扩散精细化模块后的工程代价，本节从模型大小、参数数据量与推理时间三个维度，对“仅使用 CPUNet 粗分割模型”与“CPUNet+扩散精细化模型”进行对比。实验设置如下：CPU 平台、FP32 精度、输入尺寸为 $1\\times 3\\times 256\\times 256$，扩散推理步数设置为 DDIM 20 步。

**表13-1　模型大小、参数数据量与推理时间对比**

| 指标 | CPUNet | CPUNet+扩散精细化 | 增量（后者-前者） |
|:----|------:|------------------:|-----------------:|
| 总参数量（M） | 259.31 | 340.34 | +81.03 |
| 模型大小（MB，FP32） | 989.2 | 1298.3 | +309.1 |
| 单张推理时间（ms） | 1384.1 | 38386.5 | +37002.4 |
| 推理时间倍率（×） | 1.00 | 27.7 | +26.7 |

进一步地，扩散精细化模型中的新增参数主要来自条件去噪网络，新增参数规模约为 81.03M，其中图像特征编码器约占 8.54M，其余参数来自去噪 U 型网络主体及注意力相关模块。上述结果表明，在不改变粗分割骨干网络结构的前提下，引入扩散精细化模块使模型参数规模提升约 31.2%，模型文件体积增加约 31.2%。

在推理效率方面，完整模型推理时间显著高于仅粗分割模型，其主要耗时来自多步 DDIM 反向去噪过程。由实验数据可见，扩散阶段在总推理耗时中占主导地位，提示后续工作可从采样步数压缩、蒸馏加速或轻量化去噪网络设计等方向进一步优化系统部署性能。


## 第十四节　本章小结

本章系统阐述了基于条件扩散模型的医学图像分割精细化系统中扩散模块的各核心组件，并进一步给出了模型规模与推理效率的定量对比。各功能组件的定位与设计要点概括于表14-1。

**表14-1　扩散模块各组件功能总结**

| 模　块 | 功能定位 | 核心设计要点 |
|:------:|:--------:|:-----------:|
| 高斯扩散调度器 | 噪声调度与采样管理 | 支持线性/余弦调度，DDIM 加速推理 |
| 时间步嵌入编码器 | 噪声水平感知编码 | 正弦位置编码与 MLP 变换 |
| 条件残差块 | 基础特征提取单元 | 时间步嵌入加性注入 |
| 自注意力模块 | 全局空间依赖建模 | 选择性应用于低分辨率层级 |
| PFMM | 输入端先验信息调制 | 类别感知的概率加权特征调制 |
| CFFM | 低层级条件特征融合 | 深度可分离卷积与残差连接 |
| SACM | 高层级语义注意力融合 | 通道注意力与空间注意力双路协同 |
| 多尺度图像特征编码器 | 多尺度图像先验提取 | ResNet-V2 预激活瓶颈结构 |
| WS-Former | 频率域跨模态特征交互 | 小波分解+交叉注意力+时间自适应门控 |
| 条件去噪网络 | 扩散去噪主体网络 | 多模块协同的编码器—解码器架构 |
| 扩散精细化优化器 | 端到端训练与推理封装 | 粗分割起始推理、条件增强与混合损失 |

综上，本系统通过将条件扩散模型与多源条件信息（原始图像多尺度特征、粗分割先验概率图及频率域小波表示）深度融合，构建了具备较强边界精细化能力的分割框架；与此同时，模型规模与推理开销的上升也为后续轻量化部署研究提供了明确方向。


## 参考文献

[1] HO J, JAIN A, ABBEEL P. Denoising diffusion probabilistic models[C]//Advances in Neural Information Processing Systems. 2020: 6840-6851.

[2] SONG J, MENG C, ERMON S. Denoising diffusion implicit models[C]//International Conference on Learning Representations. 2021.

[3] NICHOL A Q, DHARIWAL P. Improved denoising diffusion probabilistic models[C]//International Conference on Machine Learning. PMLR, 2021: 8162-8171.

[4] RONNEBERGER O, FISCHER P, BROX T. U-Net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2015: 234-241.

[5] WOO S, PARK J, LEE J Y, et al. CBAM: Convolutional block attention module[C]//Proceedings of the European Conference on Computer Vision. 2018: 3-19.

[6] VASWANI A, SHAZEER N, PARMAR N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems. 2017: 5998-6008.

[7] HE K, ZHANG X, REN S, et al. Identity mappings in deep residual networks[C]//European Conference on Computer Vision. Springer, 2016: 630-645.

[8] MALLAT S G. A theory for multiresolution signal decomposition: the wavelet representation[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1989, 11(7): 674-693.
