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
