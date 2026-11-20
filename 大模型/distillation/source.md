# 生成模型蒸馏方法：从回归到对抗

> 目的：把扩散/生成模型常用的蒸馏路线放在一起比较，重点讲清 **DMD** 和 **GAN 蒸馏**，以及为什么 DMD2 要把两者结合起来。  
> 适合：想把大扩散模型压成 1–4 步少步生成器，或理解 Self Forcing / CSF 里 DMD 的位置。

---

## 0. 一句话地图

- **回归蒸馏**：让 student 的输出接近 teacher，简单稳定，但多模态时容易糊。
- **渐进式蒸馏**：逐步把采样步数减半，稳，但训练周期长。
- **一致性模型 / LCM**：学习 ODE 轨迹上“同一条轨迹任意两点一致”的函数，少步生成稳定。
- **Score distillation / SDS / VSD**：用 teacher 的 score 指导学生，不需要 teacher 的样本逐步对齐，但容易过饱和、模式单一。
- **DMD**：用 `real-score − fake-score` 做分布匹配，保留细节和多模态；fake-score 是在线跟踪当前 generator 的模型。
- **GAN 蒸馏**：用一个真假裁判（discriminator）逼 student 更像真的，输出更锐利，但训练不稳定。
- **DMD2**：把 DMD 和 GAN 结合起来，分布匹配 + 对抗监督，稳定性和效果更好。

---

## 1. 蒸馏在解决什么

扩散模型通常要几十到几百步采样，部署慢。蒸馏的目标是：**把一个强 teacher 的生成能力，转移到一个少步 student 上**，同时尽量保住细节和多样性。

按监督信号可以分三类：

1. **点级监督**：让 student 的输出接近 teacher 的输出或特征（MSE、感知损失、中间层回归）。
2. **分布级监督**：让 student 的分布接近 teacher/真实分布（score matching、DMD、VSD）。
3. **对抗监督**：用 discriminator 区分 teacher/真实样本和 student 样本，逼 student 更像真的（GAN、ADD、DMD2）。

核心矛盾是：**MSE 简单但容易糊；分布匹配和 GAN 更锐利，但训练更复杂、更不稳定。**

---

## 2. 方法总览

| 方法 | 核心思想 | 监督信号 | 优点 | 主要问题 |
|---|---|---|---|---|
| **回归蒸馏** | 匹配 teacher 输出/特征 | teacher 输出、特征 | 简单、稳定、便宜 | 多模态下条件均值 → 糊 |
| **渐进式蒸馏** | 逐步把采样步数减半 | teacher 轨迹 | 稳定 | 训练慢、要多次蒸馏 |
| **一致性模型 / LCM** | 学习 ODE 轨迹的一致性 | ODE 轨迹 | 少步稳定、部署快 | 轨迹近似误差 |
| **SDS / VSD** | 用 teacher 的 score 指导学生 | teacher score | 不需要 teacher 样本逐步对齐 | 过饱和、模式单一 |
| **DMD** | real-score − fake-score 做分布匹配 | real + fake score | 细节好、多模态保留 | fake-score 在线跟踪，训练复杂 |
| **GAN / ADD** | 真假裁判做对抗 | 对抗 | 输出锐利 | 训练不稳定 |
| **DMD2** | DMD + GAN | score + 对抗 | 更稳、效果更好 | 组件更多、调参更多 |

---

## 3. 回归、渐进式、一致性模型

### 3.1 回归蒸馏

最直接的做法：让 student 的输出接近 teacher。

```text
L = || G_student(z) − G_teacher(z) ||²
```

也可以匹配中间层特征、感知特征、latent。优点是简单稳定；缺点是**多模态时最优解是条件均值**，输出容易糊。所以它适合目标比较确定、模态单一的任务，或者作为其他蒸馏方法的辅助项。

### 3.2 渐进式蒸馏

把采样步数逐步减半：

```text
teacher 64 步 → student 32 步 → student 16 步 → … → 1–4 步
```

每一步都让新 student 拟合上一步的采样结果。优点是稳定；缺点是要训很多次，流程长，误差会累积。

### 3.3 一致性模型 / LCM

学习一个“ consistency function”：同一条 ODE 轨迹上，任意两点都能映射到同一个输出。这样 student 不需要逐步模拟完整轨迹，可以 1–4 步直接出图。LCM 是这条路线在 latent diffusion 上的常用版本，工程上很成熟。

---

## 4. Score distillation / SDS / VSD

### 4.1 基本想法

teacher 是一个扩散模型，能提供 score：

```text
s_teacher(x_t, t) ≈ ∇ log p_teacher(x_t)
```

student 的生成样本加噪后，用 teacher 的 score 告诉它往哪走，让输出更接近真实分布。

### 4.2 常见问题

只用 teacher score 会有两个毛病：

- **过饱和**：所有样本都被推向高密度区，细节被抹平；
- **模式单一**：多个模态被压到一个峰上，多样性丢失。

VSD（Variational Score Distillation）引入一个可训练的 score 模型去估计当前生成分布，缓解这个问题；DMD 则把它明确写成 real-score 和 fake-score 的差。

---

## 5. DMD：Distribution Matching Distillation

### 5.1 三个角色

| 角色 | 是什么 | 作用 |
|---|---|---|
| **G（generator / student）** | 少步生成器，比如 1–4 步去噪 | 要训练的学生 |
| **real-score** | 冻结的预训练扩散模型 | 知道真实数据长什么样，指向真实分布 |
| **fake-score** | 在线更新的扩散模型 | 跟踪当前 G 的分布，给出当前生成分布的 score |

更新方向：

```text
s_real − s_fake
```

即“真实分布的方向”减去“当前生成分布的方向”。G 沿这个方向更新，fake-score 继续跟上 G。

### 5.2 score 是什么、怎么算

score 是一个和输入同维的向量场：

```text
s(x_t, t) = ∇_{x_t} log p_t(x_t)
```

扩散前向加噪：

```text
x_t = α_t x_0 + σ_t ε,    ε ~ N(0, I)
```

条件 score：

```text
∇ log p_t(x_t | x_0) = −(x_t − α_t x_0) / σ_t² = −ε / σ_t
```

对 `x_0` 取期望：

```text
s(x_t, t) = −E[ε | x_t] / σ_t
```

去噪网络 `ε_θ(x_t, t)` 学的就是 `E[ε | x_t]`，所以：

```text
s_θ(x_t, t) = −ε_θ(x_t, t) / σ_t
```

**注意**：不是除以时间 `t`，而是除以该时刻的噪声标准差 `σ_t`。

### 5.3 DMD loss 怎么算

```python
x0 = G(z)                          # 少步生成
t = randint(0, T)
eps = randn_like(x0)
x_t = alpha[t] * x0 + sigma[t] * eps

s_real = real_score(x_t, t)        # 冻结
s_fake = fake_score(x_t, t)        # 在线更新

d = s_real - s_fake                # 从当前生成分布往真实分布推
target = (x0 + d).detach()
L_G = mse(x0, target)              # 实际还会带 w(t) 权重
L_G.backward()                     # 只更新 G

L_fake = mse(fake_score(x_t, t), eps)
L_fake.backward()                  # 只更新 fake-score
```

关键是：**score 差是常数方向，梯度只穿过 G 的生成过程**。

### 5.4 为什么不能直接用 student 当 fake-score

因为 student 的 `v` 是拟合 teacher 的分布，不是拟合它自己的输出分布。

一个 1D 例子：

- 真实数据 `N(0,1)`；
- student 4 步采样实际产出 `N(0,2)`，偏糊；
- 但 student 的 `v` 仍在拟合 `N(0,1)`。

如果拿 student 当 fake：

```text
s_real ≈ score of N(0,1)
s_fake = s_student ≈ score of N(0,1)
s_real − s_fake ≈ 0
```

DMD 会觉得“已经对齐了”，不会去修正 `N(0,2)` 的方差。

如果单独训 fake-score，让它在 student 实际产出的 `N(0,2)` 样本上学习：

```text
s_fake ≈ score of N(0,2)
s_real − s_fake ≠ 0
```

它才会告诉 generator：“你的实际输出方差太大了，往 `N(0,1)` 推。”

所以：**G 负责生成，fake-score 负责测量 G 当前生成分布。** 两者不能混为一谈。

### 5.5 fake-score 怎么来

fake-score 不是从数据里“长出来”的，它是一个可训练的 diffusion/flow 模型：

1. **初始化**：从同家族预训练权重复制，常见是 generator 初始权重或同尺寸 base 模型；
2. **拿样本**：当前 generator 生成 `x̂_0`，detach；
3. **加噪**：`x_t = (1−t)x̂_0 + tε`；
4. **训练 fake**：用 flow matching / 去噪 loss 拟合噪声或速度，更新 fake-score；
5. **循环**：generator 更新后，fake-score 继续在新样本上训练，在线跟踪 `p_θ`。

它不是 GAN 的 discriminator，不判断 real/fake 标签，只给 score；推理时 fake-score 和 real-score 都丢掉，只保留 generator。

---

## 6. GAN 蒸馏

### 6.1 基本设定

- **Teacher / 真实数据**：提供真实样本；
- **Student G**：少步生成器；
- **Discriminator D**：判断“这个样本是真的还是 G 生成的”。

学生不再追求逐像素像 teacher，而是追求**让 D 分不出真假**。

### 6.2 裁判的输入

D **一次只看一张图**，不是同时看 teacher 和 student 两张图做对比：

```text
真实图 x_real  → D → D(x_real) 应该大
G 的图 x_fake  → D → D(x_fake) 应该小
```

输入通常是**最终生成的图像**，或者经过 DINOv2 / CLIP / VGG 等预训练模型提取的特征。常见做法是“冻结视觉特征 + 一个小分类头”，比像素级判别器稳定。

### 6.3 训练流程

```python
# 1) 更新 D
x_real = sample_real_data()          # 或 teacher 生成的样本
x_fake = G(z).detach()
D_loss = -log(D(x_real)) - log(1 - D(x_fake))
D_loss.backward()                     # 只更新 D

# 2) 更新 G
x_fake = G(z)
G_loss = -log(D(x_fake))              # G 希望 D 判它为真
G_loss.backward()                     # 只更新 G
```

D 和 G 交替更新：

- D 学会更会挑毛病；
- G 学会更会掩盖毛病；
- 理想结果是 D 分不出真假，G 的输出分布接近真实分布。

### 6.4 为什么不用 MSE

- MSE 是逐点匹配，未来不确定时最优解是条件均值 → 糊；
- GAN 是分布层面的对抗，**更容易保留细节和多模态**；
- 所以少步生成时，GAN 蒸馏通常比纯回归更锐利。

### 6.5 稳定训练的关键

GAN 蒸馏最大的问题是**不稳定**，常见手段：

- **feature-space discriminator**：用 DINOv2 / CLIP / VGG 冻结特征，避免脆弱的像素判别器；
- **R1 / 谱归一化 / 梯度惩罚**：限制 D 不要太强；
- **two-time-scale**：D 和 G 的学习率、更新频率分开；
- **EMA student**：评估时用 EMA 权重；
- **加蒸馏回归项**：给 G 一个稳定锚点，防止纯 GAN 漂移。

一个典型例子是 **Adversarial Diffusion Distillation（ADD，SDXL Turbo 用的方法）**：student 是 1–4 步生成器，D 在 DINOv2 特征上区分 teacher 和 student 的样本，loss 是 **adversarial + score distillation** 的组合。

---

## 7. DMD2：DMD + GAN

DMD2 不是“用 GAN 替代 DMD”，而是**在 DMD 的 score 差梯度上再加一个对抗损失**：

- DMD 部分：`s_real − s_fake` 给分布匹配方向；
- GAN 部分：D 给“像不像真的”的对抗压力。

两者互补：

- DMD 提供分布层面的梯度；
- GAN 提供更直接的真实感监督，减少模糊和过饱和，稳定训练。

DMD2 还做了：

- **two-time-scale 更新**：fake-score 更新更频繁，跟上 generator；
- **EMA generator**：用 EMA 权重做评估；
- **去掉 R1 正则**；
- 更好的噪声 schedule 和训练配置。

注意：CSF 里用的是标准 DMD，不是 DMD2 的 GAN 版本。

---

## 8. 怎么选

| 你的情况 | 建议 |
|---|---|
| 想快速验证、训练稳定 | LCM / 一致性模型 |
| 有 teacher 的清晰轨迹、允许慢慢蒸馏 | 渐进式蒸馏 |
| 目标分布比较单一、模态少 | 回归蒸馏 + 感知损失 |
| 想保留细节和多模态、能接受复杂训练 | DMD / DMD2 |
| 有真实数据但没有好的 teacher score | GAN / ADD |
| 做少步视频生成、关心长程一致性 | DMD（Self Forcing / CSF 这条线） |
| 只部署、不训练 | 直接选蒸馏好的现成模型，不用关心 DMD/GAN |

---

## 9. 常见坑

1. **MSE 蒸馏变糊**：多模态时条件均值不是真实样本，输出容易糊。
2. **SDS 过饱和**：只用 teacher score，样本被推向同一高密度区，多样性下降。
3. **fake-score 漂移**：DMD 里 fake-score 更新太快或太慢，梯度都会错；two-time-scale 很关键。
4. **GAN 不稳定**：D 太强 G 学不动，G 太强 D 失效；需要 feature-space D、EMA、梯度惩罚。
5. **teacher 不匹配**：real-score 或 teacher 必须和目标领域一致；换领域要换 teacher。
6. **指标误导**：一致性/清晰度提高可能来自运动量下降，质量和 Dynamic Degree 要一起看。
7. **部署时忘记丢辅助模型**：DMD 的 real/fake score、GAN 的 discriminator 都只在训练时用，推理只保留 generator。

---

## 参考

- DMD：Yin et al., *One-step Diffusion with Distribution Matching Distillation*
- DMD2：Yin et al., *Improved Distribution Matching Distillation for Fast Image Synthesis*
- LCM：Luo et al., *Latent Consistency Models*
- Consistency Models：Song et al., *Consistency Models*
- ADD：Sauer et al., *Adversarial Diffusion Distillation*
- Self Forcing / CSF：<https://arxiv.org/abs/2610.12156>
