# π0 / openpi 代码分析

- 仓库：Physical-Intelligence/openpi，commit 215abfb
- 模型：π0、π0-FAST、π0.5；最新是 π0.5
- π0：PaliGemma 2B + Gemma 300M action expert，flow matching
- π0-FAST：PaliGemma 2B + FAST tokenizer，自回归
- π0.5：Pi0Config(pi05=True)，离散 state + adaRMS，只支持 flow matching head

## 关键文件

```text
src/openpi/models/pi0.py
src/openpi/models/pi0_fast.py
src/openpi/models/gemma.py
src/openpi/models/siglip.py
src/openpi/training/config.py
scripts/train.py
src/openpi/policies/policy.py
src/openpi/policies/policy_config.py
src/openpi/models_pytorch/pi0_pytorch.py
```

## 输入序列怎么拼

- prefix：3×256 图像 token + 最多 48 个 prompt token = 816
- suffix（π0）：1 个 state token + 50 个 action token = 51
- 整体：816 + 51 = 867 个 token
- π0.5：state 离散进 prompt，suffix 只有 50 个 action token
- time：π0 拼到每个 action token 特征上；π0.5 用 adaRMS
- attention：prefix 不能看 suffix；state 不能看 action；action 能看 prefix、state 和所有 action

## mask

- 整个序列是 block-causal，不是标准 causal
- prefix 内部双向 full attention
- prefix 看不到 suffix
- state 能看 prefix + state，不能看 action
- action 能看 prefix + state + 所有 action
- `ar_mask` 标记 block 边界：prefix block 0，state block 1，action block 2

## 两个 expert 怎么实现

- 一个 `gemma.Module`，`configs=[paligemma_config, action_expert_config]`
- 参数名用 `_name(name, i)` 区分：`qkv_einsum` / `qkv_einsum_1`、`mlp` / `mlp_1`（expert 0 无后缀）
- 固定路由：prefix→expert 0，suffix→expert 1；没有 gate/top-k
- 共享 attention；FFN/QKV/输出投影各自独立

## 最后怎么出动作

- `suffix_out: [B, 51, 1024]`
- 取最后 50 个 action token：`[B, 50, 1024]`
- `action_out_proj` -> `v_t: [B, 50, 32]`
- 训练：`v_t` 和 `u_t = noise - actions` 做 MSE
- 推理：10 步 Euler `x_t = x_t + dt * v_t`，最后 `x_0` 是 action chunk
- 50 个 action token 并行解码，不是自回归

## 推理 10 步循环

- Prefix：PaliGemma 2B，18 层 × 1 次 = 18 层
- 去噪：Action Expert 300M，18 层 × 10 次 = 180 层
- 合计 11 次 forward，198 层执行
- Prefix K/V 只算一次并缓存；每个去噪 step 只跑 Action Expert
- 每个 step 输入 x_t 和 t，输出 v_t，Euler 更新 x_t

## 为什么这样设计

- PaliGemma 2048：保留预训练 VLM
- Action Expert 1024/300M：动作任务窄，省算力
- 共享 attention：不需要额外 cross-attention
- QKV 都投到 8×256：不同宽度也能拼
- 固定路由：prefix/suffix 天然分开，无需 gate
- prefix KV cache：2B 只跑 1 次，300M 跑 10 次

## π0.5

- `Pi0Config(pi05=True)`
- state 离散到 256 bin，拼进 prompt：`Task: ..., State: ...; Action:`
- suffix 只有 50 个 action token
- time 用 adaRMSNorm 注入 action expert
- `max_token_len=200`；prefix 968，suffix 50，total 1018
- 只支持 flow matching head
- `pi05_libero` 用 `discrete_state_input=False`，state 被忽略

## π0.6 / π0.7

- 官方 openpi 只到 π0.5，没有 π0.6 / π0.7 代码和权重
- π0.6：Gemma3 4B + 860M action expert，Knowledge Insulation，4 图 448×448，5 步去噪，RECAP（π*0.6）
- π0.7：5B，Gemma3 4B + MEM video history encoder + 860M action expert，多模态上下文 + subgoal images + BAGEL world model，5 步
- 官方资料：π0.6 model card、arXiv 2511.14759、arXiv 2604.15483
- 第三方实现：openpie-0.6、opentau pi06、hzm8341/pi0.6、CBC_Pi0.7_Openpi（非官方）
