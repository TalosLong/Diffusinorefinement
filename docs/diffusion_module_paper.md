# 基于条件扩散模型的医学图像分割优化方法：扩散模块组件详述

# Conditional Diffusion Model for Medical Image Segmentation Refinement: A Detailed Description of Diffusion Module Components

---

## 摘要 (Abstract)

本文详细描述了一种基于条件扩散模型的医学图像分割优化系统中扩散模块的各个组件。该系统采用两阶段架构：首先通过粗分割网络（CPUNet）获得初始分割结果，然后利用条件扩散模型对分割边界进行精细化优化。扩散模块包含高斯扩散调度器、条件去噪U-Net、先验特征调制模块（PFMM）、条件特征融合模块（CFFM）、空间与通道注意力模块（SACM）、图像特征编码器、以及小波空间Transformer（WS-Former）等核心组件。本文按模块逐一阐述各组件的设计原理、数学公式和实现细节。

---

## 1. 引言 (Introduction)

医学图像分割是计算机辅助诊断的基础任务。尽管深度学习方法（如U-Net及其变体）已在该领域取得显著进展，但粗分割结果往往在目标边界区域存在不精确性。为解决这一问题，本系统引入去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）作为后处理优化手段，通过逐步去噪过程精细化分割边界。

系统整体架构如下：

```
输入图像 → 预处理 → CPUNet（粗分割）→ 扩散优化模块 → 后处理 → 最终分割结果
```

扩散优化模块是本系统的核心创新点，其内部由多个精心设计的子模块组成。以下各节将分别详细描述这些组件。

---

## 2. 高斯扩散调度器 (Gaussian Diffusion Scheduler)

### 2.1 概述

高斯扩散调度器（`GaussianDiffusion` 类）是扩散模型的数学基础，负责管理前向扩散（加噪）和反向扩散（去噪）过程中的噪声调度。

### 2.2 前向扩散过程 (Forward Diffusion Process)

前向扩散过程定义了如何逐步向干净数据 $x_0$（真实分割掩码）中添加高斯噪声：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

其中：
- $\beta_t$ 为噪声调度参数（noise schedule），按线性或余弦方式递增
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ 为累积乘积

直接采样公式为：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### 2.3 噪声调度方案 (Beta Schedule)

系统支持两种噪声调度方案：

**线性调度 (Linear Schedule)：**

$$\beta_t = \beta_{\text{start}} + \frac{t}{T}(\beta_{\text{end}} - \beta_{\text{start}})$$

默认参数：$\beta_{\text{start}} = 10^{-4}$，$\beta_{\text{end}} = 0.02$，$T = 1000$。

**余弦调度 (Cosine Schedule)：**

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s = 0.008$ 为偏移常数，防止 $\beta_t$ 在 $t$ 接近 $0$ 时过小。

### 2.4 后验分布参数 (Posterior Distribution)

反向过程的后验分布 $q(x_{t-1} | x_t, x_0)$ 相关参数预计算如下：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

### 2.5 DDIM 采样 (DDIM Sampling)

为加速推理，系统采用 DDIM（Denoising Diffusion Implicit Models）采样策略。DDIM 允许在远少于 $T$ 步的子序列上进行确定性或随机采样：

给定预测的 $\hat{x}_0$ 和推导的噪声 $\hat{\epsilon}$：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \hat{\epsilon} + \sigma_t \cdot z$$

其中 $\sigma_t = \eta \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$，$\eta = 0$ 时为确定性采样。

系统默认采用 20-50 步 DDIM 采样替代完整的 1000 步采样，在推理效率和结果质量之间取得良好平衡。

---

## 3. 时间步嵌入 (Timestep Embedding)

### 3.1 概述

时间步嵌入（`get_timestep_embedding` 函数）将离散的扩散时间步 $t$ 映射为连续的高维向量表示，使去噪网络能够感知当前的噪声水平。

### 3.2 正弦位置编码

采用 Transformer 中经典的正弦位置编码方案：

$$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right)$$

$$\text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

其中 $d$ 为嵌入维度（默认 256），$i$ 为维度索引。

该编码随后经过两层 MLP 进一步变换：

$$e_t = W_2 \cdot \text{SiLU}(W_1 \cdot \text{PE}(t) + b_1) + b_2$$

输出维度与残差块内部通道数一致，用于逐层注入时间步信息。

---

## 4. 条件残差块 (Conditional Residual Block)

### 4.1 概述

条件残差块（`ConditionalResBlock` 类）是去噪U-Net的基本构建单元，在标准残差连接的基础上引入了时间步条件。

### 4.2 结构设计

每个残差块包含以下计算流程：

$$h = \text{Conv}_{3 \times 3}(\text{SiLU}(\text{GN}(x)))$$

$$h = h + \text{MLP}(e_t)$$

$$h = \text{Conv}_{3 \times 3}(\text{Dropout}(\text{SiLU}(\text{GN}(h))))$$

$$\text{output} = h + \text{Shortcut}(x)$$

其中：
- $\text{GN}$ 为分组归一化（Group Normalization，8 组）
- $\text{SiLU}$ 为 Sigmoid Linear Unit 激活函数：$\text{SiLU}(x) = x \cdot \sigma(x)$
- $e_t$ 为时间步嵌入，通过 MLP 投影后以加性方式注入
- 当输入输出通道数不一致时，$\text{Shortcut}$ 使用 $1 \times 1$ 卷积进行维度匹配

### 4.3 时间步注入机制

时间步信息通过以下方式注入特征图：

$$h' = h + (W_t \cdot \text{SiLU}(e_t))[:, :, \text{None}, \text{None}]$$

即将时间步嵌入经线性变换后扩展为空间维度，与卷积特征逐通道相加。这使得网络在不同噪声水平下表现出不同的去噪行为。

---

## 5. 自注意力块 (Self-Attention Block)

### 5.1 概述

自注意力块（`AttentionBlock` 类）用于捕获特征图中的长距离空间依赖关系，在去噪U-Net的瓶颈层和特定分辨率层中使用。

### 5.2 多头自注意力机制

给定输入特征 $X \in \mathbb{R}^{B \times C \times H \times W}$：

1. 先进行分组归一化：$\hat{X} = \text{GN}(X)$

2. 将空间维度展平：$\hat{X}_{\text{flat}} \in \mathbb{R}^{B \times (H \cdot W) \times C}$

3. 计算多头注意力（默认 4 头）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

4. 残差连接：$\text{output} = X + \text{Proj}(\text{Attention}(\hat{X}_{\text{flat}}))$

系统在分辨率为 $16 \times 16$ 和 $8 \times 8$ 的特征图上应用自注意力，以在计算效率和表征能力之间取得平衡。

---

## 6. 先验特征调制模块 (Prior Feature Modulation Module, PFMM)

### 6.1 概述

PFMM（`PFMM` 类）是扩散去噪网络的输入端调制模块。它利用 CPUNet 生成的粗分割概率图 $P$ 作为先验信息，对初始卷积后的特征 $Z$ 进行调制，使去噪网络在处理初始阶段就能感知粗分割结果的空间分布。

### 6.2 数学公式

设 $P \in \mathbb{R}^{B \times K \times H \times W}$ 为 $K$ 类概率图，$Z \in \mathbb{R}^{B \times C \times H \times W}$ 为待调制特征。

**步骤一：概率图通道拆分**

$$P = [P_1, P_2, \ldots, P_K], \quad P_k \in \mathbb{R}^{B \times 1 \times H \times W}$$

**步骤二：加权特征拼接**

$$Z' = \text{Concat}(P_1 \odot Z, P_2 \odot Z, \ldots, P_K \odot Z) \in \mathbb{R}^{B \times (K \cdot C) \times H \times W}$$

其中 $\odot$ 表示逐元素乘法（广播机制下的通道级加权）。

**步骤三：深度可分离卷积与残差连接**

$$Z'' = \text{LN}(\text{DWConv}(Z') \oplus Z)$$

其中 $\text{DWConv}$ 为深度可分离卷积（先逐通道卷积，再 $1 \times 1$ 逐点卷积），$\oplus$ 为残差加法。

**步骤四：最终输出**

$$Z''' = \text{ReLU}(\text{LN}(\text{Conv}_{3 \times 3}(Z'')))$$

### 6.3 设计动机

通过将每个类别的概率图分别与特征相乘，PFMM 实现了类别感知的特征调制。高概率区域的特征得到增强，低概率区域的特征被抑制，从而引导去噪过程关注需要精细化的区域。

---

## 7. 条件特征融合模块 (Conditional Feature Fusion Module, CFFM)

### 7.1 概述

CFFM（`CFFM` 类）负责将原始图像的低级特征注入到去噪U-Net编码器的各层中。它在编码器的前三个尺度层（高分辨率层）工作，确保去噪过程能够参考原始图像的纹理和边缘信息。

### 7.2 数学公式

设 $F \in \mathbb{R}^{B \times C_f \times H \times W}$ 为U-Net编码器某层的特征，$X_{\text{low}} \in \mathbb{R}^{B \times C_x \times H \times W}$ 为图像编码器对应尺度的低级特征（经投影对齐通道数后）。

**步骤一：特征拼接**

$$F' = \text{Concat}(F, X_{\text{low}}) \in \mathbb{R}^{B \times (C_f + C_x) \times H \times W}$$

**步骤二：深度可分离卷积处理**

$$F'' = \text{ReLU}(\text{LN}(\text{DWConv}(F')))$$

其中 $\text{DWConv}$ 将拼接后的通道映射回 $C_f$ 维度。

**步骤三：残差连接与归一化**

$$F''' = \text{LN}(F'' \oplus F)$$

### 7.3 设计动机

低级图像特征（如边缘、纹理）对于分割边界的精确定位至关重要。CFFM 通过深度可分离卷积高效融合这些信息，同时残差连接确保了编码器原有特征的保留。

---

## 8. 空间与通道注意力模块 (Spatial and Channel Attention Module, SACM)

### 8.1 概述

SACM（`SACM` 类）在去噪U-Net编码器的最深层（最低分辨率层）使用，负责将图像编码器提取的高级语义特征融入编码器的深层表示。与CFFM不同，SACM通过注意力机制（而非简单拼接）实现更精细的特征交互。

### 8.2 通道注意力 (Channel Attention, CA)

**步骤一：特征投影**

对高级图像特征 $X_{\text{high}}$ 进行卷积投影：

$$X' = \text{Conv}_{3 \times 3}(X_{\text{high}}) \in \mathbb{R}^{B \times C_s \times H \times W}$$

**步骤二：全局池化**

$$X'_{\text{max}} = \text{GlobalMaxPool}(X') \in \mathbb{R}^{B \times C_s}$$

$$X'_{\text{avg}} = \text{GlobalAvgPool}(X') \in \mathbb{R}^{B \times C_s}$$

**步骤三：通道注意力权重**

$$\text{CA} = \sigma(\text{MLP}(X'_{\text{max}}) + \text{MLP}(X'_{\text{avg}})) \in \mathbb{R}^{B \times C_s}$$

其中 MLP 为共享参数的两层全连接网络（含通道缩减比 $r = 16$）：

$$\text{MLP}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$

### 8.3 空间注意力 (Spatial Attention, SA)

**步骤一：空间级池化**

$$X'_{\text{max\_spatial}} = \max_{c}(X') \in \mathbb{R}^{B \times 1 \times H \times W}$$

$$X'_{\text{avg\_spatial}} = \text{mean}_{c}(X') \in \mathbb{R}^{B \times 1 \times H \times W}$$

经通道扩展后拼接：

$$X_{\text{concat}} = \text{Concat}(X'_{\text{max\_spatial}}, X'_{\text{avg\_spatial}}) \in \mathbb{R}^{B \times 2C_s \times H \times W}$$

**步骤二：空间注意力权重**

$$\text{SA} = \sigma(\text{DWConv}_{7 \times 7}(X_{\text{concat}})) \in \mathbb{R}^{B \times 1 \times H \times W}$$

### 8.4 特征融合

$$S' = \text{Conv}_{3 \times 3}(S)$$

$$S'' = \text{LN}(S \oplus (\text{CA} \odot S') \oplus (\text{SA} \odot S'))$$

其中 $S$ 为U-Net编码器的深层特征，$\oplus$ 为逐元素加法，$\odot$ 为逐元素乘法。

### 8.5 设计动机

SACM 借鉴了 CBAM（Convolutional Block Attention Module）的思想，但将其改造为跨模态特征融合机制。通道注意力捕获"哪些特征通道重要"，空间注意力捕获"哪些空间位置重要"，两者协同将高级语义信息精确地注入到去噪过程中。

---

## 9. 图像特征编码器 (Image Feature Encoder)

### 9.1 概述

图像特征编码器（`ImageFeatureEncoder` 类）基于 ResNet-V2 架构，从原始输入图像中提取多尺度特征，供 CFFM 和 SACM 使用。

### 9.2 架构设计

编码器采用预激活瓶颈残差块（Pre-Activation Bottleneck），结构如下：

| 阶段 | 输出通道数 | 输出分辨率 | 残差块数量 | 供给模块 |
|------|-----------|-----------|-----------|---------|
| Root（初始卷积） | 64 | $H/2 \times W/2$ | - | CFFM (Level 0) |
| Stage 1 | 256 | $H/4 \times W/4$ | 3 | CFFM (Level 1) |
| Stage 2 | 512 | $H/8 \times W/8$ | 4 | CFFM (Level 2) |
| Stage 3 | 1024 | $H/16 \times W/16$ | 6 | SACM (Level 3) |

### 9.3 预激活瓶颈残差块

每个 Pre-Activation Bottleneck 的计算流程为：

$$y = \text{ReLU}(\text{GN}_1(\text{Conv}_{1 \times 1}(x)))$$

$$y = \text{ReLU}(\text{GN}_2(\text{Conv}_{3 \times 3}(y)))$$

$$y = \text{GN}_3(\text{Conv}_{1 \times 1}(y))$$

$$\text{output} = \text{ReLU}(y + \text{Downsample}(x))$$

其中使用权重标准化卷积（`StdConv2d`）：

$$\hat{w} = \frac{w - \mu_w}{\sqrt{\sigma_w^2 + \epsilon}}$$

### 9.4 通道投影

由于图像编码器各阶段的通道数（64, 256, 512, 1024）与去噪U-Net编码器各层的通道数（64, 128, 256, 512）不完全匹配，系统引入 $1 \times 1$ 卷积投影层将其对齐：

$$\hat{F}_i = \text{ReLU}(\text{GN}(\text{Conv}_{1 \times 1}(F_i)))$$

---

## 10. 小波空间Transformer (Wavelet-Space Transformer, WS-Former)

### 10.1 概述

WS-Former（`WS_Former` 类）是本系统的核心创新模块，在去噪U-Net编码器的最深层使用（与SACM并行）。它利用离散小波变换（DWT）将特征分解到频率域，在小波空间中进行扩散特征与条件特征的交叉注意力交互，并根据时间步自适应地控制不同频率子带的权重。

### 10.2 小波分解 (Wavelet Decomposition)

对输入特征 $F \in \mathbb{R}^{B \times C \times H \times W}$ 进行一级 Haar 小波变换：

$$\text{DWT}(F) = (F_{\text{LL}}, [F_{\text{LH}}, F_{\text{HL}}, F_{\text{HH}}])$$

其中：
- $F_{\text{LL}} \in \mathbb{R}^{B \times C \times H/2 \times W/2}$：低频子带（近似系数），包含主要语义信息
- $F_{\text{LH}}, F_{\text{HL}}, F_{\text{HH}} \in \mathbb{R}^{B \times C \times H/2 \times W/2}$：高频子带（水平、垂直、对角细节系数），包含边缘和纹理信息

### 10.3 小波域交叉注意力 (Wavelet Cross-Attention)

对扩散噪声特征 $N$ 和条件特征 $C$ 分别进行小波分解后，在各子带之间进行交叉注意力：

**低频子带对齐（语义对齐）：**

$$\hat{N}_{\text{LL}} = \text{CrossAttn}_{\text{LL}}(N_{\text{LL}}, C_{\text{LL}}) + N_{\text{LL}}$$

其中 Query 来自扩散特征，Key/Value 来自条件特征：

$$Q = W_Q \cdot N_{\text{LL}}, \quad K = W_K \cdot C_{\text{LL}}, \quad V = W_V \cdot C_{\text{LL}}$$

$$\text{CrossAttn}(Q, K, V) = W_O \cdot \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**高频子带对齐（边缘/细节对齐）：**

三个高频子带（LH, HL, HH）合并到批次维度并行处理，共享同一注意力模块参数：

$$[\hat{N}_{\text{LH}}, \hat{N}_{\text{HL}}, \hat{N}_{\text{HH}}] = \text{CrossAttn}_{\text{High}}([N_{\text{LH}}, N_{\text{HL}}, N_{\text{HH}}], [C_{\text{LH}}, C_{\text{HL}}, C_{\text{HH}}]) + [N_{\text{LH}}, N_{\text{HL}}, N_{\text{HH}}]$$

### 10.4 时间自适应子带门控 (Time-Adaptive Subband Gating)

`TimeAdaptiveSubbandGating` 模块根据当前扩散时间步 $t$ 动态调整各子带的权重，核心思想是：

- **早期去噪阶段**（$t$ 较大）：噪声水平高，应更依赖低频语义信息
- **后期去噪阶段**（$t$ 较小）：噪声水平低，应更关注高频细节信息

门控权重计算：

$$[g_{\text{LL}}, g_{\text{LH}}, g_{\text{HL}}, g_{\text{HH}}] = \sigma(\text{MLP}(e_t)) \in [0, 1]^{C \times 4}$$

各子带加权：

$$\hat{N}'_{\text{LL}} = g_{\text{LL}} \odot \hat{N}_{\text{LL}}$$

$$\hat{N}'_{\text{LH}} = g_{\text{LH}} \odot \hat{N}_{\text{LH}}, \quad \hat{N}'_{\text{HL}} = g_{\text{HL}} \odot \hat{N}_{\text{HL}}, \quad \hat{N}'_{\text{HH}} = g_{\text{HH}} \odot \hat{N}_{\text{HH}}$$

### 10.5 小波重构与输出

经门控加权后，通过逆小波变换（IDWT）重构空间域特征：

$$F_{\text{out}} = \text{IDWT}(\hat{N}'_{\text{LL}}, [\hat{N}'_{\text{LH}}, \hat{N}'_{\text{HL}}, \hat{N}'_{\text{HH}}])$$

最终通过 MLP 投影（$1 \times 1$ 卷积 + BatchNorm + SiLU + $1 \times 1$ 卷积）得到输出。

### 10.6 设计动机

传统扩散模型在空间域直接处理特征，难以同时兼顾全局语义一致性和局部边缘精度。WS-Former 通过将交互过程转移到小波域实现频率解耦，低频子带负责语义对齐，高频子带负责边缘细节对齐，时间自适应门控则确保去噪过程中的频率关注点随噪声水平动态变化。

---

## 11. 条件去噪U-Net (Conditional Denoising U-Net)

### 11.1 概述

条件去噪U-Net（`ConditionalDenoisingUNet` 类）是扩散模块的主体网络架构，基于U-Net编码器-解码器结构，综合集成了上述各条件注入模块。

### 11.2 整体架构

```
输入（噪声掩码 x_t）
    │
    ▼
初始卷积 (Conv 3×3) → base_channels
    │
    ▼
PFMM 调制（引入粗分割先验 P）
    │
    ▼
╔═══════════════════════════════════╗
║          编码器 (Encoder)          ║
║                                   ║
║  Level 0: 2×ResBlock + CFFM      ║ ← 图像低级特征 (64 ch, H/2)
║      ↓ Downsample                 ║
║  Level 1: 2×ResBlock + CFFM      ║ ← 图像低级特征 (256 ch, H/4)
║      ↓ Downsample                 ║
║  Level 2: 2×ResBlock + CFFM      ║ ← 图像低级特征 (512 ch, H/8)
║      ↓ Downsample                 ║
║  Level 3: 2×ResBlock + Attn      ║
║           + SACM + WS-Former     ║ ← 图像高级特征 (1024 ch, H/16)
╚═══════════════════════════════════╝
    │
    ▼
╔═══════════════════════════════════╗
║       瓶颈层 (Bottleneck)         ║
║  ResBlock → Attention → ResBlock  ║
╚═══════════════════════════════════╝
    │
    ▼
╔═══════════════════════════════════╗
║          解码器 (Decoder)          ║
║                                   ║
║  Level 3: 3×ResBlock (+ skip)    ║
║      ↑ Upsample                  ║
║  Level 2: 3×ResBlock (+ skip)    ║
║      ↑ Upsample                  ║
║  Level 1: 3×ResBlock (+ skip)    ║
║      ↑ Upsample                  ║
║  Level 0: 3×ResBlock (+ skip)    ║
╚═══════════════════════════════════╝
    │
    ▼
输出层 (GN → SiLU → Conv 3×3)
    │
    ▼
预测 logits (B, C_out, H, W)
```

### 11.3 网络参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_channels` | 64 | 基础通道数 |
| `channel_mult` | (1, 2, 4, 8) | 各层通道倍数 |
| `num_res_blocks` | 2 | 每层残差块数量 |
| `attention_resolutions` | (16, 8) | 应用注意力的分辨率 |
| `time_emb_dim` | 256 | 时间嵌入维度 |
| `dropout` | 0.1 | Dropout 比率 |
| `input_resolution` | 256 | 输入图像分辨率 |

通道数变化：编码器 64 → 128 → 256 → 512，解码器对称上采样。

### 11.4 条件注入策略总结

| 编码器层级 | 分辨率 | 图像特征来源 | 注入模块 | 注入方式 |
|-----------|--------|-------------|---------|---------|
| 输入层 | $H \times W$ | 粗分割概率图 | PFMM | 先验调制 |
| Level 0 | $H/2$ | ResNet Root (64ch) | CFFM | 拼接+DWConv |
| Level 1 | $H/4$ | ResNet Stage1 (256ch) | CFFM | 拼接+DWConv |
| Level 2 | $H/8$ | ResNet Stage2 (512ch) | CFFM | 拼接+DWConv |
| Level 3 | $H/16$ | ResNet Stage3 (1024ch) | SACM + WS-Former | 注意力+小波 |
| 所有层 | - | 时间步 $t$ | ResBlock 内 MLP | 加性注入 |

---

## 12. 扩散优化器 (Diffusion Refiner)

### 12.1 概述

扩散优化器（`DiffusionRefiner` 类）是最顶层的封装模块，整合了冻结的 CPUNet 粗分割网络、高斯扩散调度器和条件去噪U-Net，提供端到端的训练和推理接口。

### 12.2 训练过程

训练阶段的计算流程如下：

1. **获取粗分割掩码**：$P = \text{softmax}(\text{CPUNet}(I))$

2. **条件增强**（50% 概率应用）：
   - 随机形态学操作（膨胀/腐蚀），模拟推理时的粗分割误差
   - 添加小幅高斯噪声扰动：$P' = \text{clamp}(P + \mathcal{N}(0, 0.05^2), 0, 1)$

3. **准备真值**：将 Ground Truth 掩码 $M$ 缩放到 $[-1, 1]$：$x_0 = 2M - 1$

4. **前向扩散**：随机采样时间步 $t \sim \text{Uniform}(0, T)$，添加噪声：$(x_t, \epsilon) = q(x_0, t)$

5. **去噪预测**：$\hat{x}_0 = f_\theta(x_t, t, I, P)$（网络直接预测 $x_0$ 的 logits）

6. **损失计算**：

$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

其中：

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_i[M_i \log(\hat{p}_i) + (1 - M_i)\log(1 - \hat{p}_i)]$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i \hat{p}_i M_i + \epsilon}{\sum_i \hat{p}_i + \sum_i M_i + \epsilon}$$

### 12.3 推理过程

推理阶段采用 DDIM 采样策略：

1. **获取粗分割掩码**：$P = \text{softmax}(\text{CPUNet}(I))$

2. **缩放至扩散空间**：$P_{\text{scaled}} = 2P - 1$

3. **从中间时间步开始**：默认 $t_{\text{start}} = 0.7T$，对粗分割掩码添加对应噪声水平的噪声

4. **DDIM 反向采样**：在 20-50 个等间距子时间步上逐步去噪

5. **缩放回概率空间**：$\hat{M} = \text{clamp}((x_0 + 1) / 2, 0, 1)$

### 12.4 设计动机

从粗分割结果（而非纯噪声）出发进行去噪是本系统的关键设计选择。这种策略确保了：
- 整体分割结构得以保留（粗分割已捕获主要形态）
- 去噪过程集中于边界细化和噪声消除
- 推理效率显著提高（仅需从 $0.7T$ 而非 $T$ 开始去噪）

---

## 13. 总结 (Summary)

本文详细描述了基于条件扩散模型的医学图像分割优化系统中扩散模块的各个组件。各模块的协作关系总结如下：

| 模块 | 角色 | 关键创新点 |
|------|------|-----------|
| 高斯扩散调度器 | 噪声调度管理 | 支持线性/余弦调度，DDIM 加速采样 |
| 时间步嵌入 | 噪声水平感知 | 正弦编码 + MLP 变换 |
| 条件残差块 | 基础特征提取 | 时间步条件加性注入 |
| 自注意力块 | 长距离依赖建模 | 选择性应用于低分辨率层 |
| PFMM | 先验信息注入 | 类别感知的特征调制 |
| CFFM | 低级特征融合 | 深度可分离卷积+残差的高效融合 |
| SACM | 高级语义融合 | 双路注意力（通道+空间） |
| 图像特征编码器 | 多尺度特征提取 | ResNet-V2 预激活瓶颈 |
| WS-Former | 频率域特征交互 | 小波分解+交叉注意力+时间门控 |
| 条件去噪U-Net | 主体去噪网络 | 多模块集成的编码器-解码器 |
| 扩散优化器 | 端到端封装 | 粗分割起始+条件增强+混合损失 |

该系统通过将扩散模型与丰富的条件信息（原始图像多尺度特征、粗分割先验、频率域表示）相结合，实现了高质量的医学图像分割边界优化。

---

## 参考文献 (References)

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
2. Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR*.
3. Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *ICML*.
4. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
5. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional block attention module. *ECCV*.
6. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. *ECCV*.
8. Mallat, S. (1989). A theory for multiresolution signal decomposition: The wavelet representation. *IEEE PAMI*.
