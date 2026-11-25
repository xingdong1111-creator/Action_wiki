# Cosmos 3 原理学习：统一全模态、MoT 与 Flow Matching

资料范围：NVIDIA 官方论文《Cosmos 3: Omnimodal World Models for Physical AI》arXiv:2606.02800v4（2026-06-23），以及用户提供的知乎文章摘录。本文只讨论公开论文，不涉及任何公司内部实现。公式与配置以官方 v4 为准，直觉解释用于辅助理解。

> 💡 **整篇论文的核心可以压缩成一句话：**把语言和视觉理解放进 AR 区，把视频、音频、动作等连续生成目标放进 DM 区；每层用 Reasoner / Generator 两套参数处理，再用统一的 clean/noisy 排列表达不同任务。


## Cosmos 3 想统一什么

过去，视觉语言理解、视频生成、世界模拟和动作策略往往由不同模型完成。Cosmos 3 希望用同一套主干覆盖这些能力，让输入输出的组合决定任务，而不是每换一种任务就换一套架构。

| 能力 | 典型输入 | 典型输出 |
|---|---|---|
| 视觉语言模型 | 语言、图像、视频 | 文本回答、定位、时序与动作理解 |
| 媒体生成器 | 文本，可选初始图像、视频或控制信号 | 图像、视频、音频 |
| 世界模拟器 | 当前观测与动作 | 动作之后的未来世界状态 |
| 世界-动作模型 | 任务、当前观测与本体状态 | 动作，以及动作可能造成的视觉后果 |

支持的模态是语言、图像、视频、音频和动作。语言通过 next-token prediction 生成；其余连续模态通过迭代去噪生成。统一的代价是要解决四件事：不同模态怎样编码、token 怎样排列、两套生成目标怎样共存、不同采样率怎样对齐时间。


## 不同模态怎样进入统一表示空间

每种模态先经过自己的编码器，再投影到 Transformer 的表示空间。非语言模态还会加一个可学习的模态 embedding，让模型区分视觉、音频和动作。

| 模态 | 编码方式 | token 去向 | 训练要点 |
|---|---|---|---|
| 语言 | 语言 tokenizer | AR / Reasoner | next-token prediction |
| 理解用图像 / 视频 | 经过视觉语言对齐的 ViT，patch=16；再用 MLP 合并 2×2 token，并使用 DeepStack | AR / Reasoner | ViT 与主干联合训练 |
| 生成用图像 / 视频 | Wan2.2-TI2V-5B 视频 VAE | DM / Generator | VAE 冻结；latent 再用线性层投影到 hidden dimension |
| 音频 | 48 kHz 音频 VAE，hop=1920，因此为 25 tokens/s | DM / Generator | VAE 冻结 |
| 动作 | 领域特定的输入 / 输出投影，不使用 VAE | DM / Generator | 投影从零初始化，与主干联合优化 |

**视觉有两套编码器，不能混淆：**ViT 负责“理解画面里有什么”，Wan VAE 负责“把画面压缩成可生成、可重建的连续 latent”。同为视觉 token，理解走 AR，生成走 DM。

Wan VAE 时间压缩 4 倍、空间压缩 32×32；空间压缩由 16×16 的 VAE 下采样和 2×2 patch merge 共同完成。因此后文的一个视频 latent frame 对应四个原始视频帧。


### 动作如何统一不同机器人与控制接口

论文用共享的几何组件表达动作：ego / effector 的相对位姿由 3D 平移和 6D 旋转组成，grasp state 直接表示当前抓取状态。不同 embodiment 的动作长度和语义不必相同，各自使用领域特定的输入输出投影：

```text
输入：z = W_in^(k)  x + b_in^(k)
输出：x = W_out^(k) z + b_out^(k)
k = embodiment / domain 标识
x = 原始动作向量
z = MoT 共享表示空间中的动作 embedding
```

预测的 6D 旋转再通过 SVD 转回 3×3 的 SO(3) 旋转矩阵。统一的是共享语义空间，不是强迫所有机器人使用同一种原始控制向量。


## Token 怎样排列：AR 在前，DM 在后

所有任务都可以写成一条序列：

```text
S = [ S_AR ; S_DM ]
S_AR = 语言 token + 可选 ViT 图像/视频 token + EOS + BOG
S_DM = VAE 视觉 token + 音频 token + 动作 token
EOS = end-of-sentence
BOG = begin-of-generation
```

排列只有三条硬规则：

1. AR 子序列永远在 DM 子序列前。
1. DM 内，每种模态的 clean conditioning tokens 排在 noisy diffusion tokens 前。
1. 条件段和目标段内部，都按 vision → audio → action 排序。

**clean / noisy 不是“数据干不干净”：**

- clean：已知的、没有施加当前生成噪声的条件 latent；它参与 attention，但不计入去噪损失。
- noisy：待生成目标与噪声插值后的 latent；模型需要从它恢复 clean target。

因此，任务切换很大程度上就是改变 clean 和 noisy 的位置，而不必改变网络结构。


## MoT：一条序列，两套逐层参数

Mixture-of-Transformers 的每个 decoder layer 都有两条完整 pathway：

- Reasoner tower：处理 AR 子序列。
- Generator tower：处理 DM 子序列。

两条路径各自拥有独立的 LayerNorm、attention projection matrices 和 FFN，都从预训练 VLM 权重初始化。token 属于 AR 还是 DM 决定了它走哪套参数。

> 💡 **MoT 不是常见的稀疏 MoE。**这里没有 learned router、top-k 专家选择或负载均衡；路由由 token 所在的子序列确定。两塔也不是互不相干的两个模型，它们在每个 decoder layer 的 joint attention 中交互。


### AR：保持自回归因果注意力

```text
O_AR = Attn_causal(Q_AR, K_AR, V_AR)
```

AR token 只能读取 AR 内部的前序 token，不能读取 DM。这保留了 VLM 的自回归理解和生成能力。


### DM：双向读取 AR 与全部 DM

```text
O_DM = Attn_full(
    Q_DM,
    [K_AR ; K_DM],
    [V_AR ; V_DM]
)
```

DM token 可以读取文本等 AR 条件、clean 条件和其他 noisy target，从而联合保持时空一致性。关键的不对称是：**生成可以读取理解，理解不会被生成反向改写。**

“共享 self-attention operator”不等于两塔共享 Q/K/V 投影权重；投影仍是各自独立的。


## 一套 clean/noisy 模板表达所有生成任务

记号：v / s / a 分别表示 clean 的视觉、音频、动作；带波浪号的对应符号表示 noisy target。

| 模式 | clean 条件 | noisy / 待生成目标 | 任务含义 |
|---|---|---|---|
| Text-to-Image | AR 文本；无 clean DM 图像 | 目标图像 | 文本生成一张图，视作单帧视频 |
| Text-to-Video | AR 文本；无 clean DM 视频 | 全部目标视频，可选音频 | 从文本生成整段媒体 |
| Image-to-Video | 文本 + 第一个 clean latent frame | 后续视频 | 让初始图像动起来 |
| Video-to-Video | 文本 + 前 P 个 clean latent frames | 第 P+1 帧之后的视频 | 从历史前缀续写未来；预训练常用 P=2 |
| Video Transfer | 文本 + 整段 clean control video | 对应的 RGB 视频 | 把 depth、edge 等控制视频转换成目标视频 |
| Forward Dynamics | 观测上下文 + clean action | 未来视觉 | 给定动作，预测动作后果 |
| Inverse Dynamics | clean vision，描述已观察到的视觉变化 | 动作 | 反推什么动作导致了视觉变化 |
| Policy | 任务、当前观测与本体状态 | 未来视觉与动作，两边都加噪 | 联合生成干预动作及其预期后果 |

动作 token a_t 表示从 v_(t-1) 到 v_t 的状态转移。V2V continuation 与 Transfer 的区别也很清楚：前者的 clean 是同一视频的历史开头，后者的 clean 可以是覆盖整段时长的控制信号。


## Flow Matching：Generator 到底学什么

对任一连续模态，clean target 是 x_0，高斯噪声是 ε，噪声水平为 σ：

```text
x_σ = σ ε + (1 - σ) x_0
v*  = ε - x_0
σ = 0 时：x_σ = x_0，干净目标
σ = 1 时：x_σ = ε，纯噪声
```

Generator 读取 x_σ、σ 和条件 c，学习预测速度 v*。训练使用 masked MSE：只有 noisy target 计入损失，clean 条件虽然参与 attention，但不作为重建目标。


### 为什么推理能一步步走回干净结果

对插值式关于 σ 求导，就得到 v* = ε - x_0。生成从 σ=1 走向 σ=0，因此 σ_next < σ。用 Euler 离散化：

```text
x_(σ_next) = x_σ + (σ_next - σ) v_θ(x_σ, σ, c)
```

注意符号：带符号的 Δσ<0 时用加号；如果把正步长写成 Δ=σ-σ_next>0，公式就等价地写成 x_next=x-Δv。

**一个标量例子：**x_0=2，ε=-1，σ=0.5，则 x_σ=0.5，真实速度 v*=-3。如果模型完全准确，走到 σ_next=0：

```text
x_next = 0.5 + (0 - 0.5) × (-3) = 2 = x_0
```

真实模型不会每步都准确，所以需要多个去噪步逐步逼近。


### 不同模态的噪声安排

- 各模态独立采样 σ，不是整样本共用一个噪声水平。
- image / audio / action 使用 logit-normal；video 使用 mode sampling。
- mid-training 中 action loss 权重为 10，补偿归一化动作向量较小的逐元素 MSE；这不是把动作数值乘 10。
- 论文还写到 action 继承 vision noise schedule，但这不等于 action 与 video 共用同一个随机 σ。


### Noise shift：让训练更关注高噪声区域

```text
σ = s t̄ / (1 + (s - 1) t̄)，其中 t̄ = 1 - t，s ≥ 1
```

s 越大，噪声分布越偏向高噪声区域。它改变的是训练时噪声水平的分布，不是视频 FPS，也不是把 latent 乘一个常数。

| 训练阶段 | 256p | 480p | 720p |
|---|---|---|---|
| Generator pre-training | s=1 | s=3 | s=5 |
| Generator mid-training | s=3 | s=5 | s=10 |


## 3D MRoPE：把不同模态放到同一物理时间轴

每个 token 获得时间 t、高度 h、宽度 w 三个位置坐标：

- 语言：t=h=w，同步递增，退化为 1D RoPE。
- ViT 媒体 token：同帧共享 t，h/w 表示空间位置。
- DM 视频 token：t 随 latent frame 前进，h/w 在每帧的空间网格变化。
- 音频和动作：只使用时间坐标，h=w=0。

视觉 segment 开始时坐标会重置，因此这里的 t/h/w 是视频内部的绝对坐标，不是整条 packed sequence 的全局 token 序号。


### AR 与 DM 之间的 15000 gap

论文发现，DM 直接紧接 AR 的时间位置会导致初始视频帧过饱和和棋盘格伪影，Super 上尤其明显。解决办法是把后续视觉、音频和动作的时间索引统一平移，固定留下 15000 的位置间隔。

**15000 是位置坐标 gap，不是新增 15000 个 token，也不是 15000 帧或 15000 秒。**论文对其病因的解释是：最后一个语言 token 与第一帧视觉 token 的时间 embedding 过于接近；gap 提供更清晰的文本到视觉转换信号。


### TPS 与 FPS modulation

TPS 表示每秒 temporal steps。视频经过 VAE 的 4 倍时间压缩，所以：

```text
TPS_video  = FPS / 4
TPS_audio  = 48000 / 1920 = 25
TPS_action = 动作采样频率
TPS_base   = 24 / 4 = 6
每个局部时间步真正加入 MRoPE 的增量：
δt = TPS_base / TPS
```

| 模态 / 采样率 | TPS | δt |
|---|---|---|
| 24 FPS 视频 | 6 | 1 |
| 60 FPS 视频 | 15 | 0.4 |
| 16 FPS 视频 | 4 | 1.5 |
| 48 kHz、hop=1920 音频 | 25 | 0.24 |
| 15 Hz 动作 | 15 | 0.4 |

经过相同的一秒，任何模态都满足 TPS × δt = 6。因此它们在位置轴上走过相同的长度。论文最终采用 Text Control + MRoPE FPS Modulation，提示中也会加入 duration 和 FPS。


## 把机制串起来：两个端到端例子


### Image-to-Video

1. 文本进入 AR；初始图像经 Wan VAE 变成第一个 clean latent frame，放在 DM 前部。
1. 未来视频 latent 加噪，放在 clean prefix 后。
1. Reasoner 处理 AR 条件；Generator 在每个去噪步读取 AR、clean prefix 和当前 noisy future，预测视频 velocity。
1. sampler 多次更新未来 latent，clean prefix 保持不变，也不计入训练去噪损失。
1. 最终只把未来视频 latent 交给 Wan VAE decode，得到新视频帧。


### Policy：动作与未来视频一起生成

1. 任务文本、当前观测和本体状态作为条件。
1. 未来动作和未来视频都初始化为噪声，联合放进 DM。
1. Generator 每步同时预测动作 velocity 和视频 velocity，二者通过 attention 共同约束。
1. 去噪结束后，动作反归一化并执行；视频 latent 可以解码用于分析，也可以在低延迟控制时跳过 decode。

这也是世界模型与策略模型能在同一架构里会合的原因：动作既可以是 clean 条件（FD），也可以是 noisy 目标（ID / Policy）。


## 模型规模与训练课程

| 变体 | 整体标称参数 | dense 基础 | 初始化与发布状态（论文 v4） |
|---|---|---|---|
| Cosmos3-Edge | 4B | 2B | 底层 LLM 从零训练；后续发布 |
| Cosmos3-Nano | 16B | 8B | Qwen3-VL 8B 架构与预训练权重；已发布 |
| Cosmos3-Super | 64B | 32B | Qwen3-VL 32B 架构与预训练权重；已发布 |

整体参数约为同级 dense 模型的两倍，正是因为每层有 Reasoner 与 Generator 两套路径。


### Reasoner：先学会理解

- 预训练约 22.0M 样本：image-text、video-text、纯文本对话，学习通用视觉语言理解。
- Physical AI SFT 约 2.2M 样本，其中 video-text 约占一半，引入机器人、自动驾驶、空间与时序理解。
- 总计约 24.2M 样本；目标是 next-token prediction。


### Generator：从 Reasoner 权重出发学习生成

1. Pre-training： 图像、视频、音频，覆盖 T2I / T2V / I2V / V2V；Reasoner 冻结，只更新 generation-specific 参数。使用 74K token packing、10% text dropout和多分辨率课程。
1. Mid-training： 加入 action、video transfer 和更多 Physical AI 数据，学习可控世界动态。Action 数据约 8.4M episodes、61.3K hours。
1. Post-training： 架构不变，特化为 Text2Image、Image2Video、Policy-DROID 等专家模型。

论文后半部分还给出完整工程体系：数据筛选与标注、HSDP + Context Parallelism、在线 Wan VAE、异步 checkpoint、基准追踪，以及 vLLM / TensorRT-LLM / vLLM-Omni 推理服务。这些支撑规模化训练与部署，但不是理解 MoT 的前置条件。


## 推理优化：哪些缓存不是一回事

| 机制 | 缓存 / 并行什么 | 性质 |
|---|---|---|
| 常规 AR KV cache | 自回归文本生成时的前序 token K/V | 标准自回归解码优化 |
| Reasoner tower caching | 采样开始前算一次 AR 条件输出，之后所有 denoise steps 复用 | 由于 AR 不读取 DM，论文明确为数学等价、不影响质量 |
| Cache-DiT | 复用相邻 Generator denoising steps 的部分 block outputs | training-free 近似优化，论文称质量影响可忽略；不同于 Reasoner cache |
| CFG parallelism | conditional / unconditional 两次 forward 放到不同 GPU | 降低每个 denoise step 的延迟 |
| Context parallelism | 把长媒体 token 序列切到多个 GPU | 降低单卡显存与长序列延迟 |

Reasoner 还可以先做 prompt upsampling：把简短输入扩写成结构化场景描述，再交给 Generator。它属于提示工程与条件构建，不是 MoT 必须执行的固定步骤。


## 最容易误解的十个点

1. MoT 不是 MoE： 按 AR/DM 确定性路由，不是稀疏专家选择。
1. 双塔不是两个独立模型串行调用： 每层都有两套参数，并在 joint attention 中交互。
1. ViT 与 Wan VAE 不互相替代： 前者用于理解，后者用于生成。
1. DM full attention 不等于自回归文本解码： 连续目标在一次 denoise forward 内双向联合建模。
1. clean 条件不参与去噪损失： 但它仍通过 attention 影响目标学习。
1. attention mask 与 loss mask 不同： 一个决定能看什么，一个决定算什么损失。
1. 动作不一定总是输出： FD 中动作是 clean 条件；ID / Policy 中才是待生成目标。
1. 15000 不是 token 数： 只是 AR 与 DM 之间的位置坐标间隔。
1. TPS 不是原始 FPS： 视频还要除以 VAE 时间压缩率 4。
1. 统一模型不等于物理正确性自动有保证： 物理一致性、动作准确性和任务成功率仍需对应基准验证。


## 建议的论文阅读顺序与自测

第一遍按 §2.1 编码器 → §2.2 token 排列 → §2.3 MoT → §2.4 位置编码 → §4.2 Generator Training 阅读。数据、基础设施和评测可以在架构清楚后再看。

如果能不看正文回答下面六题，说明核心心智模型已经建立：

1. 为什么理解用 ViT，生成却用 Wan VAE？
1. AR 为什么不能让 DM 反向更新自己？这带来什么缓存机会？
1. FD、ID、Policy 的 clean/noisy 分配分别是什么？
1. 从 x_σ 与 v* 的公式出发，Euler 更新的符号为什么容易写错？
1. 24 FPS 视频与 15 Hz 动作为什么在一秒后都走过 6 个位置单位？
1. 只改变 clean/noisy mask，为什么就能从 I2V 切换到 Inverse Dynamics？


## 补充实例：同一个 I2V 的训练与推理

用“文本：车辆向左转 + 一张初始图片 + 两个未来 latent 帧”把全流程串起来。这里只展示标准 Image-to-Video，不把不同生成模式混在一起。


### 推理：从噪声逐步生成

1. 文本进入 AR；初始图片经 Wan VAE 得到 clean v1；未来 x2、x3 从高斯噪声开始。
1. 序列为 [AR 文本 ; DM clean v1 ; DM noisy x2, x3]。
1. 每次 forward 预测 x2、x3 的 velocity，sampler 按 x_new=x_old+(σ_new-σ_old)×velocity 更新。
1. 文本和 v1 在每一步都作为条件使用，但文本理解结果/KV与 v1 只需计算一次并缓存；每步变化的是 noisy latent 和 σ。
1. 得到干净 v2、v3 后，与 v1 一起交给 Wan VAE decode。


### 训练：真实未来加噪，一次监督

1. 训练数据同时提供文本、初始图片和真实未来视频，因此 v1、v2、v3 都已知。
1. 随机采样一个视频模态的 σ，以及每个目标位置独立的高斯噪声 ε2、ε3。
1. 构造 x2=σ ε2+(1-σ)v2、x3=σ ε3+(1-σ)v3。
1. 只做一次 forward，目标是 ε2-v2 和 ε3-v3；只对 noisy future 计算 MSE，文本和 v1 不算生成损失。
1. 不同训练步覆盖不同 σ，不需要在训练时跑完整采样链。

**核心区别：**训练是“真实未来 → 加噪 → 一次 forward 学 velocity”；推理是“纯噪声 → 多次 forward → 逐步生成未来”。


## 补充细节：attention、噪声与长视频


### AR/DM 的最小 attention mask

以 [A1, A2, D1, D2] 为例，行是 query，列是可见 key：

```text
      A1   A2   D1   D2
A1    ✓    ✗    ✗    ✗
A2    ✓    ✓    ✗    ✗
D1    ✓    ✓    ✓    ✓
D2    ✓    ✓    ✓    ✓
```

AR 是带对角线的下三角因果区域；DM 在论文通用形式中是一整块 full attention。这里说的“整块”是 mask 矩阵结构，不是视频时间上的 block-causal。

标准 pre-norm decoder 可简化理解为：LN → self-attention（Q/K/V 投影、joint attention、输出投影）→ 残差 → LN → FFN → 残差。之前列 Reasoner / Generator 参数时，是在说明两条路径拥有哪些权重，不是说 FFN 在 attention 前执行。


### 标准 I2V 不是传统 teacher forcing，也不是逐帧独立 σ

真实首帧只作为 clean 条件，未来整段 latent 一起做 Flow Matching；并不是逐帧“喂真实前一帧、预测下一帧”。标准 Generator Training 按模态采样 σ，同一视频目标共享一个 σ，但每个位置的 ε 独立。也就是“噪声强度相同，噪声数值不同”，不是 Diffusion Forcing 式的逐帧独立 σ。


### 长视频不是无限长一次出

一个窗口内，所有未来 latent 联合进行多轮去噪；但窗口仍受 token budget、attention 成本、显存和训练长度限制。更长视频可使用 V2V continuation：把前一段末尾 latent 作为 clean 前缀，再生成下一段。论文预训练使用 74K token packing，256p/480p 最多 400 帧，720p 最多 300 帧。


## 与 Wan 对比：真正变化不是“一次出”

Wan 的视频生成同样可以对整段 noisy latent 联合去噪，并通过多步 sampler 生成视频。因此“全序列”和“一次出”不是 Cosmos 独有的核心差异。

```text
Wan：文本编码器 → 视频 DiT 扩散生成 → VAE decode
Cosmos 3：Reasoner（AR 理解路径）
         + Generator（DM 扩散 / Flow 生成路径）
         → 同一序列、每层 joint attention 中协作
```

Cosmos 并没有抛弃扩散生成路径，而是把它纳入 MoT。更本质的新增点是：理解与生成统一、动作成为核心模态、FD/ID/Policy 用 clean/noisy 模板统一，以及用物理时间编码对齐不同采样率。


## 公开应用：用 WSM4 控制驾驶场景生成

NVIDIA 开源的 World Scenario Map（WSM）展示了 Video Transfer 在自动驾驶中的一种落地方式。WSM4 是结构化驾驶场景的“可视化控制语言”，不是 Cosmos 的输出，也不是普通演示图。

```text
RDS-HQ 场景数据
  （地图、动态物体轨迹、交通灯、相机内外参、时间戳）
  → cosmos-auto-wm 渲染器（version=v4）
  → 每个相机的 WSM4 控制视频
  → Cosmos3-Nano-Transfer-Auto / Sim-Auto
  → 对应的 RGB 驾驶视频
```


### 目的：编辑世界状态，再生成车辆可能看到的画面

例如把“前车始终直行”改为“前车第 2 秒切入自车车道”，重新渲染 WSM4 后，Cosmos 按新条件生成前车切入的 RGB 视频。它不是直接在原 RGB 上修图，而是根据修改后的结构化场景重新生成。

这可用于构造加塞、急刹、行人横穿等长尾场景，做反事实实验，并补充自动驾驶训练与评测数据，减少重复上路采集成本。生成结果的物理合理性和任务有效性仍需单独验证，不能把模型当作严格的几何或物理仿真器。


### 物体和地图元素如何编码

| 元素 | WSM4 表示 |
|---|---|
| Car / Truck / Pedestrian / Cyclist / Others | 不同类别颜色的 3D cuboid，前深后浅渐变，投影遵循 F-theta 镜头畸变 |
| Crosswalk | 紫色填充多边形 |
| Waitline | 绿色 polyline |
| Poles | 紫色 polyline |
| 交通灯 | 颜色表示状态，三角形表示方向 |
| STOP / KEEP CLEAR | 不同颜色的填充区域，不绘制文字 |
| 道路边界、路沿、墙体 | 按类别使用不同颜色的线条 |

颜色和符号是模型输入协议，不是可以任意更换的美化方案。文档明确指出，改变控制视频配色需要用新配色重新训练。当前 Auto 模型采用 WSM 4.0.1（version=v4）；v4.1 另外增加 occupancy voxels，不是同一版本的不同写法。

公开文档：[Cosmos Framework：WSM4 rendering — Objects and other map features](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-inference-simulation-auto/scripts/wsm/docs/wsm4.md#objects-and-other-map-features)


## 补充：语言输出与图像在 AR / DM 中的位置


### 语言从哪里输出

语言由 Reasoner 的文本输出头（LM head）产生，采用 next-token prediction。以“图片 + 图里有几辆车？”为例，图像用于理解时先经过 ViT，因此图片和问题都在 AR：

```text
[ ViT 图片 token ; 问题文本 token ; 新生成的回答 token ]
```

问题可以看到排在自己前面的 ViT 图片 token。AR 最后一个位置的 hidden state 经过 LM head 得到下一个 token，再把新 token 接回 AR：

```text
[图片 token ; 问题 ; “有”]       → 预测“2”
[图片 token ; 问题 ; “有 2”]     → 预测“辆”
[图片 token ; 问题 ; “有 2 辆”]   → 预测“车”
```

所以文本是在 AR 输入序列后面逐 token 续写的，不是在 DM 中生成。AR 看不到 DM，指的是不能读取用于生成的 VAE latent；它仍然可以读取同在 AR 中的 ViT 视觉 token。


### 图生视频时，文字、原图和未来视频各在哪里

```text
[ AR：文字 token + EOS + BOG ;
  DM clean：原始图片的 Wan VAE latent v1 ;
  DM noisy：待生成视频的 noisy latent x2、x3… ]
```

- 文字在 AR，告诉模型要生成什么。
- 原始图片在 DM clean，提供视频起点，不加噪、不更新。
- 待生成视频在 DM noisy，从噪声开始，多步去噪后得到未来 latent。

原图和未来视频都能读取 AR 文字，未来视频还能读取原图和其他 DM token；AR 不读取 DM。标准 I2V 的原图不需要经过 ViT，因为目标是生成连续 latent，而不是回答图像问题。可以记成：理解图片用 ViT → AR，以图片为起点生成视频用 Wan VAE → DM clean。


## 补充：为什么支持不同长度的视频


### 模型没有固定长度的视频输出层

Generator 对每个 latent 位置分别预测 velocity，同一套参数可以处理不同数量的 token：

```text
未来 2 帧：v1 + noisy x2、x3
           → 输出 x2、x3 的 velocity
未来 4 帧：v1 + noisy x2、x3、x4、x5
           → 输出 x2、x3、x4、x5 各自的 velocity
```

视频变长只是参与计算的 token 更多，不需要更换输出层。位置编码根据当前视频的 t/h/w 坐标计算，attention mask 和 loss mask 也随样本长度构造，所以未来几帧就监督几帧。


### 训练时也使用可变长度

论文使用可变帧数、分辨率和长宽比训练。256p / 480p 预训练最多 400 帧，720p 最多 300 帧；Cosmos3-Super-Image2Video 后训练的目标是 480p、189 帧、约 24 FPS，约 8 秒。输入长度不必和训练样本完全相同，但实际范围仍受 token budget、attention 成本、显存和训练分布限制。

更长视频可以用 V2V continuation 分段续写，但可能累积漂移。FPS 也不能只当作播放速度修改：同样 189 帧，24 FPS 约 7.9 秒，12 FPS 约 15.8 秒，物理时间和动作节奏不同，推理必须设置真实 FPS。


## 补充：Token packing 是什么，为什么需要它

Token packing 是把不同样本的真实 token 序列按边界直接接起来，而不是 padding 到最长样本。例如样本 A 有 3 个 token，样本 B 有 5 个：

```text
Padding：
[A1 A2 A3 PAD PAD]
[B1 B2 B3 B4 B5]
共 10 个位置，其中 2 个 PAD 是浪费
Token packing：
[A1 A2 A3 | B1 B2 B3 B4 B5]
共 8 个真实 token，并记录 A / B 的序列边界
```

变长训练时，每条样本先独立进行 VAE 编码、加噪以及 clean/noisy mask 构造，再按边界 packing。模型一次 forward 对所有 noisy 位置预测 velocity，loss 仍按各自样本分别计算，A 和 B 不会跨样本串数据。

packing 的主要目的不是让模型看见更多视频，而是提高多样本 batch 训练的效率：减少 padding 浪费、让 batch token 数更稳定、控制显存并减少形状变化带来的重新编译。论文按约 74K token 的预算组合样本。它不是压缩视频，也不是把不同样本拼成一条语义连续的视频。


## 补充：Wan VAE 的一个 latent 帧对应几个视频帧

Wan VAE 是约 4 倍时间压缩，但不是把每 4 帧简单平均或独立打包。时间卷积会混合相邻帧的信息，而且首帧通常单独处理。常见关系为：

```text
视频帧数 = 1 + 4 × (latent 帧数 - 1)
```

| 原始视频帧 | latent 帧 |
|---|---|
| 1 | 1 |
| 5 | 2 |
| 9 | 3 |
| 13 | 4 |

因此可以记“约 4 个视频帧对应 1 个 latent 帧”，但严格理解是 VAE 的时间压缩和上下文建模，而不是每 4 帧互不相干地一一映射。


## 资料来源

- 官方论文： Cosmos 3: Omnimodal World Models for Physical AI，arXiv v4
- 用户推荐文章： 知乎 Cosmos 3 文章 ；本文结合用户粘贴的文章摘录整理。
- 官方代码： github.com/nvidia/cosmos
- 官方模型集合： huggingface.co/collections/nvidia/cosmos3
- 论文与摘录核验日期：2026-11-19。模型发布状态与榜单结论均以论文 v4 的时间口径为准。
