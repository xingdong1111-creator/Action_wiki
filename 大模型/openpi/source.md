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
