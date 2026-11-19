# 3D U-Net 与 LLM：从零理解与结合方式

> 目标：搞清楚 3D U-Net 是什么、为什么这样设计、代码怎么写，以及它和 LLM 到底能怎么结合。
>
> 先给结论：**3D U-Net 不是 LLM 的一部分，LLM 也不是 3D U-Net 的一部分。** 两者结合，本质上是在解决“1D 语言序列”和“3D 体素网格”之间的接口问题。常见结合方式有五种：文本条件注入、特征对齐、LLM 当 decoder/reasoner、LLM 当 agent 调工具、LLM + 3D 生成模型。

---

## 目录

1. U-Net 是什么
2. 3D U-Net 是什么
3. 3D U-Net 的最小代码
4. 3D U-Net 用来做什么
5. LLM 与 3D U-Net 的五种结合方式
6. 关键设计问题
7. 一个可跑的最小实验
8. 论文与资源
9. 学习路线
10. 常见误区

---

## 1. U-Net 是什么

### 1.1 2D U-Net

U-Net（Ronneberger et al., 2015）最初是医学图像分割网络。结构像一个“U”：

```
输入 H×W
  │
  ├─ Encoder：Conv → Down → Conv → Down → ... → bottleneck
  │                                              │
  └─ Decoder：Up → Concat(skip) → Conv → Up → ... → 输出 H×W
```

三个关键点：

1. **Encoder-Decoder**：encoder 不断下采样，扩大感受野，提取语义；decoder 不断上采样，恢复分辨率。
2. **Skip connection**：把 encoder 同尺度的特征直接拼到 decoder，保留边缘、纹理等细节。
3. **多尺度**：不同层级对应不同大小的物体，适合分割。

为什么有效：分割需要“这是什么”（语义）和“在哪里”（位置）。Encoder 负责语义，skip 负责位置，decoder 把两者融合。

### 1.2 3D U-Net

3D U-Net（Çiçek et al., MICCAI 2016）就是把 2D 的卷积、池化、上采样全部换成 3D：

```
输入 D×H×W（或 C×D×H×W）
  │
  ├─ Conv3d(3×3×3) → MaxPool3d(2×2×2) → ...
  │
  └─ ConvTranspose3d(2×2×2) → Concat(skip) → Conv3d → ...
```

和 2D U-Net 的区别：

| 维度 | 2D U-Net | 3D U-Net |
|---|---|---|
| 输入 | H×W | D×H×W |
| 卷积 | Conv2d 3×3 | Conv3d 3×3×3 |
| 下采样 | 2×2 | 2×2×2 |
| 上采样 | 2×2 | 2×2×2 |
| 显存 | 低 | 高一个维度 |
| 典型数据 | 2D 图像 | CT/MRI 体积、视频、体素、点云 voxel |

**原论文的一个细节**：医学体素常常是各向异性的（层内分辨率高、层间分辨率低，比如 CT 的 z 轴间距远大于 x/y）。所以 3D U-Net 并不总是用 3×3×3，也会用 3×3×1 的卷积核，尤其在首尾层。这是“3D”与“各向同性 3D”的区别。

### 1.3 形状怎么变

假设输入 `(B, C, D, H, W)`，每次 `MaxPool3d(2)`：

```
D×H×W → D/2×H/2×W/2 → D/4×H/4×W/4 → D/8×H/8×W/8
```

上采样再逐级恢复。**要求 D、H、W 能被 2 的层数次方整除**，否则 concat 时尺寸对不上。常见做法是 pad 或 resize 到 16 的倍数。

### 1.4 计算量与显存

3D 卷积的计算量大致是 2D 的 `D` 倍，显存也是。一个 `(1, 1, 128, 128, 128)` 的输入，单层 32 通道 3×3×3 卷积就很吃显存。

常用省显存手段：

- **Patch training**：不从整个体积训练，随机裁 `(64, 64, 64)` 或 `(96, 96, 96)` 的 patch。
- **Mixed precision**：AMP。
- **Gradient checkpointing**：用时间换显存。
- **Anisotropic kernel**：`3×3×1` 减少 z 方向计算。
- **Downsample early**：第一层就 stride 2，或者用更小的 base channel。
- **Sliding window inference**：推理时切 patch，重叠拼接。

### 1.5 常见变体

| 变体 | 关键改动 | 用途 |
|---|---|---|
| V-Net | 残差块 + Dice loss | 3D 医学分割 |
| Attention U-Net | skip 上加注意力门 | 抑制无关区域 |
| nnU-Net | 自动配置 2D/3D/cascade | 医学分割强基线 |
| SwinUNETR | Transformer + U-Net 结构 | 3D 医学分割 |
| ResUNet | 残差连接 | 更深的 3D 网络 |
| 3D U-Net + cross-attention | 在 bottleneck 注入文本/条件 | 条件生成、文本分割 |

---

## 2. 3D U-Net 的最小代码

下面是一个可以直接跑的最小 3D U-Net（输入 `B×1×D×H×W`，输出 `B×num_classes×D×H×W`）：

```python
import torch
import torch.nn as nn


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv3d(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=32):
        super().__init__()
        self.inc = DoubleConv3d(in_ch, base)
        self.d1 = Down3d(base, base * 2)
        self.d2 = Down3d(base * 2, base * 4)
        self.d3 = Down3d(base * 4, base * 8)
        self.u3 = Up3d(base * 8, base * 4)
        self.u2 = Up3d(base * 4, base * 2)
        self.u1 = Up3d(base * 2, base)
        self.out = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)      # D, H, W
        x2 = self.d1(x1)      # D/2, H/2, W/2
        x3 = self.d2(x2)      # D/4, H/4, W/4
        x4 = self.d3(x3)      # D/8, H/8, W/8
        x = self.u3(x4, x3)   # D/4, H/4, W/4
        x = self.u2(x, x2)    # D/2, H/2, W/2
        x = self.u1(x, x1)    # D, H, W
        return self.out(x)


if __name__ == "__main__":
    model = UNet3d(in_ch=1, out_ch=3, base=16)
    x = torch.randn(2, 1, 64, 64, 64)
    y = model(x)
    print(x.shape, "->", y.shape)  # (2,1,64,64,64) -> (2,3,64,64,64)
```

几个容易踩的点：

- `ConvTranspose3d` 的 `kernel_size=2, stride=2` 正好把尺寸翻倍；如果输入是奇数，会差 1，需要 `output_padding=1` 或 pad。
- `BatchNorm3d` 在 batch 很小时不稳定，医学分割常用 `InstanceNorm3d` 或 `GroupNorm`。
- 3D 数据增强：随机翻转、旋转 90°、弹性变形、gamma、噪声；不要做会破坏解剖结构的强仿射。
- 损失：CE + Dice 是常见组合；类别极不平衡时用 Dice 或 Tversky。

---

## 3. 3D U-Net 用来做什么

### 3.1 医学体积分割

最经典：CT/MRI 的器官、肿瘤、病灶分割。

- 数据：BraTS（脑肿瘤 MRI）、BTCV（腹部多器官 CT）、MSD（Medical Segmentation Decathlon）。
- 输出：每个体素的类别。
- 评价：Dice、IoU、HD95（Hausdorff distance）。

### 3.2 3D 感知 / occupancy / BEV

自动驾驶里，3D U-Net 或其变体常用于：

- 体素级 occupancy 预测（哪些 voxel 被占据）；
- BEV 特征图的分割/回归；
- 多视角特征 lift 到 3D 后的融合。

这类任务输入可能是多相机特征体、LiDAR voxel、或者 BEV 栅格。3D U-Net 负责“局部细节 + 多尺度上下文”。

### 3.3 视频/体积生成中的 VAE / decoder

视频生成模型（包括世界模型）常用 3D VAE 把视频压到 latent，再用 diffusion/flow 在 latent 上生成。3D U-Net 结构可以出现在：

- VAE 的 encoder/decoder（3D 卷积 + 下/上采样）；
- 视频分割/补全；
- 体素/NeRF 特征的解码。

注意：**视频生成的主力通常是 3D VAE + DiT/UNet-based diffusion**，不是“纯 3D U-Net 直接生成视频”。3D U-Net 更多是 backbone 或 decoder 的一部分。

### 3.4 3D 形状生成

- 体素表示：3D U-Net 做 coarse shape 生成或 refinement；
- 点云 voxelization 后做 3D 卷积；
- 和 diffusion 结合：在低分辨率体素上用 3D U-Net 做 denoiser，再上采样。

### 3.5 3D 重建/补全

- 稀疏视角 → 体积重建；
- 深度图融合 → TSDF/occupancy；
- 医学图像的超分/去噪/配准（3D U-Net 作为 backbone）。

---

## 4. LLM 与 3D U-Net 的五种结合方式

### 4.0 先分清两个世界

| | LLM | 3D U-Net |
|---|---|---|
| 输入 | token 序列 (B, L) | 体素网格 (B, C, D, H, W) |
| 基本算子 | attention / FFN | Conv3d / pooling / upsampling |
| 位置 | 1D 位置编码 | 3D 空间结构（隐式在卷积里） |
| 输出 | 下一个 token / 文本 embedding | 体素级特征/分割/生成 |
| 强项 | 语义、推理、指令理解 | 空间定位、局部细节、密集预测 |

所以“结合”不是把 LLM 塞进卷积，而是找一个接口，让语言信息进入 3D 网络，或者让 3D 特征进入 LLM。

### 4.1 模式 A：文本条件注入（最常用）

**做法**：用冻结的文本编码器（CLIP/T5/LLM）把指令编码成 `(B, L, C_text)`，在 3D U-Net 的某个位置注入。

三种注入方式：

1. **Cross-attention**：在 bottleneck 或 decoder 的每一层，用 3D 特征做 query，文本 embedding 做 key/value。
2. **FiLM**：用文本 embedding 生成逐通道的 `scale` 和 `shift`，调制 3D 特征：`x = x * (1 + gamma) + beta`。
3. **AdaLN**：diffusion 里常见，把文本/时间条件注入 LayerNorm 的仿射参数。

代码示意（bottleneck cross-attention）：

```python
class CrossAttn3d(nn.Module):
    def __init__(self, dim, text_dim, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(text_dim, dim * 2)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, text):
        # x: (B, C, D, H, W), text: (B, L, C_text)
        B, C, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)      # (B, N, C)
        q = self.q(self.norm(tokens))
        k, v = self.kv(text).chunk(2, dim=-1)      # (B, L, C)
        attn, _ = self.attn(q, k, v, need_weights=False)
        tokens = tokens + attn
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).view(B, C, D, H, W)
```

**优点**：实现简单，文本编码器可以冻结；适合“文本引导分割/生成”。
**缺点**：只在 bottleneck 注入，空间定位弱；全分辨率 cross-attention 太贵；文本和 3D 位置的对齐需要额外设计。

### 4.2 模式 B：特征对齐 / CLIP-style

**做法**：3D U-Net 的 encoder 输出一个全局 3D 特征，文本编码器输出文本特征，做对比学习（InfoNCE）。推理时，文本特征和 3D 局部特征算相似度，得到 zero-shot 分割/定位。

```
3D volume → 3D U-Net encoder → global feature ─┐
                                               ├─ contrastive loss
text     → text encoder      → text feature  ─┘
```

**优点**：可以 zero-shot、开放词汇；不需要为每个类别训分类头。
**缺点**：需要大量 3D-text 配对数据；全局对齐不等于像素级 grounding；细粒度定位通常还要再加一个 decoder。

这类似 3D 版的 CLIP。医学里的 BiomedCLIP、3D-LLM/PointLLM 的视觉-语言对齐都属于这一类思路。

### 4.3 模式 C：LLM 作为 decoder / reasoner

**做法**：3D U-Net 先把体积压成特征/token，投影到 LLM 的 embedding 空间，再让 LLM 结合文本做推理/回答/输出结构化结果。

```
3D volume → 3D U-Net encoder → 3D tokens → projector → LLM → 文本/框/掩码
```

或者反过来：

```
文本/场景图 → LLM → latent code → 3D U-Net decoder → 体积
```

**优点**：LLM 的语义和推理能力强；适合 VQA、报告生成、指令理解。
**缺点**：3D token 数量大，LLM 上下文吃不消；需要 3D 位置编码；空间分辨率在 tokenization 时会丢；训练数据难。

代表工作：3D-LLM、PointLLM、Chat-3D、LEO、LL3DA。它们多用点云/多视角特征，但“3D encoder → projector → LLM”的模式是一样的。

### 4.4 模式 D：LLM 作为 agent/planner，3D U-Net 作为工具

**做法**：LLM 不直接处理 3D 数据，而是理解用户指令，决定调用哪个 3D U-Net 工具（分割肝脏、检测肿瘤、生成某区域），再把结果组合成回答。

```
用户：帮我看看这个 CT 里肝脏有没有病灶
LLM → 调用 liver_seg_3dunet(volume) → 调用 lesion_seg_3dunet(liver_region)
    → 汇总：病灶位置、大小、置信度
```

**优点**：模块化、可解释、易调试；每个 3D U-Net 可以单独训练/替换。
**缺点**：错误会传播；工具接口和返回值设计很关键；LLM 看不到原始体素，只能看工具输出。

这是医疗 copilot、具身 agent 里最现实的结合方式：**LLM 做编排，3D U-Net 做感知/分割**。

### 4.5 模式 E：LLM + 3D 生成模型

**做法**：LLM 负责文本理解、场景图/布局生成，3D U-Net（或 3D diffusion/VAE）负责形状/体素生成。

```
文本 → LLM → 场景图/布局/材质描述 → 3D 生成模型（3D U-Net decoder / diffusion）→ 3D 资产
```

**优点**：利用 LLM 的世界知识和组合能力；适合 text-to-3D、场景生成。
**缺点**：3D 生成质量、几何一致性、评估都是难点；LLM 和 3D 生成器通常分开训练，联合优化难。

注意：现在 text-to-3D 的主流不是“3D U-Net 直接出体素”，而是 NeRF/Gaussian/ mesh + diffusion，3D U-Net 更多出现在 VAE/decoder 或 coarse stage。

### 4.6 五种模式对比

| 模式 | 语言怎么进 3D | 3D 怎么进 LLM | 训练难度 | 典型任务 |
|---|---|---|---|---|
| A 文本条件注入 | cross-attn / FiLM / AdaLN | 不需要 | 中 | 文本引导分割、条件生成 |
| B 特征对齐 | 对比学习共享空间 | 全局/局部特征 | 中高 | zero-shot 分割、开放词汇 |
| C LLM decoder | 3D token → projector → LLM | 3D tokens | 高 | 3D VQA、报告生成、grounding |
| D LLM agent | LLM 调 3D U-Net 工具 | 工具返回值 | 低中 | 医疗 copilot、具身任务 |
| E LLM + 3D 生成 | 文本/场景图 → 3D 生成器 | 不一定 | 高 | text-to-3D、场景生成 |

---

## 5. 关键设计问题

### 5.1 3D 位置编码

LLM 的 token 有 1D 位置编码；3D U-Net 的特征有 3D 结构。把 3D 特征送进 LLM 时，必须告诉 LLM “这个 token 在体积的哪里”。

常见做法：

- 3D 正弦/余弦位置编码，按 `(d, h, w)` 三轴编码后拼接；
- 可学习的 3D position embedding；
- 把 3D 特征按扫描顺序 flatten，再加 1D 位置编码（会丢 3D 邻接）；
- 用 3D rotary position embedding（3D RoPE），在 attention 里注入三轴位置。

没有 3D 位置编码，LLM 只能看到“一堆特征”，无法做空间 grounding。

### 5.2 分辨率与 token 数

一个 `128×128×128` 的体素，如果每个 voxel 一个 token，就是 200 万 token，LLM 根本吃不下。

常见压缩：

- 3D U-Net 下采样 8 倍 → `16×16×16 = 4096` token；
- 只取 bottleneck 特征；
- 用 query token（learnable queries）做 cross-attention，固定输出 64/128 个 token；
- 分 patch，每个 patch 一个 token；
- 只对感兴趣区域（ROI）做高分辨率。

### 5.3 显存与计算

- 3D U-Net 本身已经吃显存；再加 cross-attention 会更贵。
- 优先在 bottleneck 或低分辨率层做 cross-attention，不要在全分辨率做。
- 用 `F.scaled_dot_product_attention` / FlashAttention。
- 文本 token 通常只有几十个，cross-attention 的 KV 很小，瓶颈在 3D 的 query 数量。
- 训练时冻结文本编码器，只训 3D U-Net + projector，能省很多显存。

### 5.4 数据

3D-text 配对数据比 2D 少得多。常见来源：

- 医学报告 + CT/MRI（报告里描述病灶位置、大小）；
- 3D 资产 + 描述（ShapeNet/Objaverse 的 caption）；
- 自动驾驶场景 + 语言指令（nuScenes + 描述）；
- 合成数据（程序生成 3D 形状 + 模板化文本）。

如果数据少，先做**模板化文本 + 合成数据**，验证接口和训练流程，再上真实数据。

### 5.5 训练策略

推荐顺序：

1. **只训 3D U-Net**：用分割/重建任务预训练，确保 3D 特征可用。
2. **冻结 3D U-Net，训 projector + LLM adapter**：对齐语言和 3D 特征。
3. **解冻部分 3D U-Net 层 + LoRA**：联合微调。
4. **全量联合训练**：数据量足够时再考虑。

不要一上来端到端全量训练，容易崩。

### 5.6 评测

- 分割：Dice、IoU、HD95。
- Grounding：文本指代区域的 IoU、命中率。
- 生成：Chamfer distance、FID、CLIP score、人工评分。
- VQA/报告：准确率、BLEU/ROUGE、临床指标、人工。
- Agent：任务成功率、工具调用次数、延迟。

---

## 6. 一个可跑的最小实验

目标：让 3D U-Net 根据文本生成/分割一个简单的 3D 形状。

### 6.1 数据

程序生成 `32×32×32` 的体素：

- 形状：cube / sphere / cylinder；
- 颜色/属性：red / blue / green（用不同通道或类别表示）；
- 文本模板：`"a red cube"`, `"a blue sphere"`, `"a green cylinder"`。

任务 A（分割）：输入体素 + 文本，输出该形状的 mask。
任务 B（生成）：输入文本 + 噪声，输出体素（可以用 3D U-Net 做 denoiser）。

### 6.2 模型

```python
# 伪代码
text_emb = frozen_clip_text(prompt)          # (B, L, 512)
volume = volume.float().unsqueeze(1)         # (B, 1, 32, 32, 32)

# 3D U-Net encoder
x1 = inc(volume)
x2 = d1(x1)
x3 = d2(x2)
x4 = d3(x3)                                  # bottleneck

# 在 bottleneck 注入文本
x4 = cross_attn(x4, text_emb)

# decoder
x = u3(x4, x3)
x = u2(x, x2)
x = u1(x, x1)
mask = out(x)                                # (B, 1, 32, 32, 32)
```

### 6.3 训练

- Loss：BCE + Dice。
- 优化器：AdamW，lr 1e-4。
- Batch：8–16。
- 训练 20–50 epoch，单卡即可。
- 数据增强：随机旋转 90°、翻转、平移。

### 6.4 对比实验

| 设置 | 目的 |
|---|---|
| 无文本，只输入体素 | baseline：模型能不能只靠形状猜 |
| 文本 + FiLM | 最简单的条件注入 |
| 文本 + bottleneck cross-attention | 本文方法 |
| 文本 + 每层 cross-attention | 看空间条件是否更强 |
| 错误文本（"blue sphere" 但输入 red cube） | 看模型是否真的听文本 |

### 6.5 预期结果

- 无文本 baseline 只能学到先验，遇到同形状不同颜色会错；
- FiLM 能区分全局属性，但空间定位弱；
- cross-attention 在“按文本选形状/颜色”上更好；
- 错误文本实验能暴露模型是否忽略语言。

这个实验很小，但能把“3D 条件注入”的整个链路跑通。

---

## 7. 论文与资源

### 7.1 3D U-Net 基础

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
- Çiçek Ö, Abdulkadir A, Lienkamp S S, Brox T, Ronneberger O. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. MICCAI 2016.
- Milletari F, Navab N, Ahmadi S A. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV 2016.
- Isensee F, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 2021.
- Hatamizadeh A, et al. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. 2022.
- Oktay O, et al. Attention U-Net: Learning Where to Look for the Pancreas. 2018.

### 7.2 3D + 语言

- Hong Y, et al. 3D-LLM: Injecting the 3D World into Large Language Models. NeurIPS 2023.
- Xu R, et al. PointLLM: Empowering Large Language Models to Understand Point Clouds. 2023.
- Wang Z, et al. Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes. 2023.
- Huang J, et al. LEO: An Embodied Generalist Agent in 3D World. ICML 2024.
- Chen S, et al. LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning. CVPR 2024.
- Zhang S, et al. BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. 2023.

### 7.3 3D 生成

- Poole B, et al. DreamFusion: Text-to-3D using 2D Diffusion. ICLR 2023.
- Lin C-H, et al. Magic3D: High-Resolution Text-to-3D Content Creation. CVPR 2023.
- Wang T, et al. Rodin: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion. CVPR 2023.

### 7.4 数据与代码

- BraTS: https://www.med.upenn.edu/cbica/brats/
- BTCV: https://www.synapse.org/#!Synapse:syn3193805
- MSD: http://medicaldecathlon.com/
- ShapeNet: https://shapenet.org/
- Objaverse: https://objaverse.allenai.org/
- MONAI: https://monai.io/ （3D U-Net、数据加载、增强）
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet
- PyTorch3D: https://pytorch3d.org/

---

## 8. 学习路线（4 周）

### 第 1 周：2D → 3D

- 手写 2D U-Net，理解 encoder/decoder/skip。
- 改成 3D U-Net，跑通 `(1,1,64,64,64) → (1,3,64,64,64)`。
- 观察参数量、显存、感受野。

### 第 2 周：真实 3D 分割

- 选 BraTS 或 MSD 的一个小任务。
- 实现 patch training、Dice loss、滑动窗口推理。
- 跑一个 baseline，记录 Dice。

### 第 3 周：文本条件

- 用 CLIP 文本编码器 + FiLM，做合成形状的文本条件分割。
- 换成 cross-attention，对比效果。
- 做“错误文本”实验。

### 第 4 周：LLM 结合

- 读 3D-LLM / PointLLM，画清楚它们的 3D encoder → projector → LLM 流程。
- 用一个小 LLM（Qwen2.5-0.5B/1.5B）做 projector + 文本输出，跑一个 toy 3D VQA。
- 或者做模式 D：LLM 调 3D U-Net 工具，做一个“医学报告生成”小 demo。

---

## 9. 常见误区

1. **“3D U-Net 是 3D 版的 Transformer”**：不是。它仍是卷积网络，强项是局部特征和密集预测，不是长程推理。
2. **“LLM 可以直接处理体素”**：不能。必须经过 tokenizer/projector/工具接口，把 3D 变成 1D token 或结构化输出。
3. **“把 3D 特征 flatten 给 LLM 就行”**：会丢 3D 邻接和位置，token 数也爆炸。需要 3D 位置编码和压缩。
4. **“3D U-Net + LLM 一定要端到端训练”**：不一定。模式 D（LLM 调工具）最省事；模式 A（条件注入）可以冻结文本编码器；端到端最贵。
5. **“3D 分割 Dice 高就说明语言理解好”**：不一定。可能是数据集先验/类别不平衡。要做错误文本、反事实、zero-shot 测试。
6. **“3D U-Net 只能做分割”**：它也可以做去噪、重建、VAE、occupancy、生成 backbone。
7. **“体素分辨率越高越好”**：显存和 token 数会爆炸；先用低分辨率验证方法，再上高分辨率。

---

## 10. 一句话总结

- **3D U-Net** = 在 3D 体素上做“下采样-上采样 + skip”的卷积网络，强项是密集预测和空间定位。
- **LLM** = 1D 序列上的语义/推理模型，强项是语言理解和组合。
- **结合** = 在两者之间建接口：文本条件注入（A）、特征对齐（B）、LLM 当 decoder（C）、LLM 当 agent 调工具（D）、LLM + 3D 生成（E）。
- **最难的不是网络结构，而是**：3D 位置编码、token 压缩、3D-text 数据、空间 grounding 的评测。

---

*笔记版本：2026-11-18。建议先跑第 2 节的 3D U-Net，再做第 6 节的 toy 实验；不要一开始就上大模型和大体素。*
