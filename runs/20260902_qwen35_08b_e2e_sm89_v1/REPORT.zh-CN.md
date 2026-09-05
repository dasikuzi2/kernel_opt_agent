# Qwen3.5-0.8B 在 RTX 4060 Laptop 上的端到端优化可行性报告

## 结论

本次验证证明了两件不同的事：

1. 这台 8GB RTX 4060 Laptop 能运行并研究 Qwen3.5-0.8B 的真实生产推理路径。vLLM 自动选择了 Triton/FLA GDN prefill、CUDA GDN decode 与 FlashAttention 2；三种 workload 的 128 个输出 token 均与 Transformers BF16 reference 完全一致。
2. 在“同一 BF16 权重、batch=1、单 token 自回归、禁止量化/推测解码/跳层”的严格赛道里，通用的 2 倍目标仍不成立，但 vLLM 并非每个形状都已最优。针对 SM89 上 `M=1, N=248320, K=1024` 的 BF16 `lm_head`，专用 Triton GEMV 在相邻 3 warmup × 10 trial 验证中达到 **1.180x / 1.184x**；随后又在同一份源码、同一编译缓存、仅切换运行时开关的 C-S-S-C 复验中达到 **1.193x E2E / 1.199x TPOT**。两轮六类自然请求的 128-token 输出均逐 token 相等。

所以，“比朴素 Transformers 快很多倍”已经实现；“在这个具体低并发形状上进一步快过 stock vLLM 约 18%”也已实现；“严格 BF16 下普遍再快 2 倍”仍不能成立。若业务目标必须是 2 倍，需要显式进入第二赛道：降低每 token 的权重字节数（量化）、一次权重读取产出多个有效 token（推测/MTP）或改变模型/硬件。

## 冻结对象

- 模型：`Qwen/Qwen3.5-0.8B`，ModelScope master 下载。
- 模型配置 SHA256：`b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`。
- 权重 SHA256：`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`。
- 权重大小：1,746,942,600 bytes；Range 下载逐段校验 `Content-Range`，最终 SHA256 与 ModelScope linked ETag 一致。
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，compute capability 8.9，8GB。
- vLLM 环境：vLLM `0.28.1rc1.dev312+g41848caa6`、Torch `2.13.0+cu130`、Triton `3.7.1`、Transformers `5.16.1`。
- 执行约束：BF16、batch=1、concurrency=1、text-only、greedy、prefix cache 关闭、每个请求状态/KV cache 新建、max model length 4096。

模型共有 873,438,784 个参数。视觉部分 100,592,896 参数被 language-model-only 模式排除。严格生成路径中纳入带宽下界的活跃权重为 1,541,502,656 bytes/token，分布如下：

| 部分 | 存储字节 |
|---|---:|
| Embedding / tied LM head | 508,559,360 |
| 18 层 Gated DeltaNet | 379,562,688 |
| MLP | 550,502,400 |
| 6 层 full attention | 102,767,616 |
| Normalization | 110,592 |

## 正确性与性能

### 逐 token 正确性

vLLM 的每个 case 先在 3 次 discovery 测量中检查重复一致性，再由独立 Transformers BF16 路径生成完整 128 token 对照。三个 case 均为精确相等，首个不一致位置均为 `null`。

### vLLM discovery baseline

本轮采用 1 次 warmup、3 次计时，case 顺序按 iteration 轮换。它足以做候选排序和可行性判断，但还不是 3 warmup、10 trial 的最终 qualification。

| Workload | 中位 TTFT | 中位 TPOT | 输出速度 | 中位 E2E |
|---|---:|---:|---:|---:|
| prompt 128 / generate 128 | 26.71 ms | 6.497 ms | 153.92 tok/s | 851.89 ms |
| prompt 512 / generate 128 | 29.92 ms | 6.893 ms | 145.07 tok/s | 904.79 ms |
| prompt 2048 / generate 128 | 113.60 ms | 6.895 ms | 145.03 tok/s | 988.67 ms |

### 与 Transformers 慢路径的对照

Transformers 当前缺少 `flash-linear-attention` 与 `causal-conv1d`，因此该数据只代表朴素 reference，不是成熟 serving baseline。

| Workload | Transformers | vLLM | vLLM 加速 |
|---|---:|---:|---:|
| prompt 128 / generate 128 | 7.665 s | 0.852 s | 9.00x |
| prompt 512 / generate 128 | 6.665 s | 0.905 s | 7.37x |
| prompt 2048 / generate 128 | 7.144 s | 0.989 s | 7.23x |

这说明模型级“大幅提升”主要来自选对生产 runtime、专用 GDN kernel、编译和 CUDA Graph。不能把这部分收益再次记到自研算子名下。

## 4060 持续候选搜索（第二轮）

第二轮没有把原始 Transformers 当作优化起点，而是直接以成熟 vLLM BF16 路径为对手。候选覆盖运行时专用化、GDN decode 后端、chunked prefill、编译 custom-op 策略和 FP8 KV cache。所有成功运行的候选都与冻结 vLLM baseline 的三组完整 128-token 输出逐 token 相等。

为降低实验闭环延迟，模型另复制到 WSL EXT4：同一 1.63 GiB checkpoint 的权重加载从 9P 上的 17.42 秒降到 EXT4 热缓存下的 0.33--1.23 秒。这个收益只减少启动等待，不改变 GPU 稳态推理。

当前同一时段的主要 discovery 结果如下。加权值使用冻结 workload 的 0.2/0.3/0.5 权重；不同 engine 进程间尚未做交错 paired qualification，所以小于约 2% 的差异均视为噪声，不作胜出声明。

| 候选 | trials | 加权 E2E | 加权 TPOT | 初始化 | 决策 |
|---|---:|---:|---:|---:|---|
| CUDA GDN、max_num_seqs=1、512 MiB 固定 cache | 7 | 961.44 ms | 6.957 ms | 25.60 s | 保留为快速实验配置 |
| Triton GDN decode、其余相同 | 7 | 1014.18 ms | 7.290 ms | 24.93 s | 淘汰 |
| CUDA GDN、max_num_seqs=80、512 MiB 固定 cache | 7 | 959.76 ms | 6.922 ms | 31.06 s | 延迟差异不确定，启动更慢 |
| 关闭 chunked prefill | 3 | 968.15 ms | 7.003 ms | 22.84 s | vLLM 明确警告该模型不正式支持；淘汰 |
| 强制全部 custom ops | 3 | 963.10 ms | 6.957 ms | 97.88 s（首次编译） | 无可测收益；淘汰 |
| max_num_seqs=1、自动显存 profile | 3 | 972.20 ms | 7.028 ms | 26.41 s | 固定 cache 的热启动收益仅 1.03x |

明确结论：

- 该轮 CUDA fused GDN 相比 Triton GDN 的加权 E2E 快 1.055x、TPOT 快 1.048x；后续带遥测的低频复验显示二者可落入 1% 内，因此这个 5% 排名只对当轮状态成立。
- max_num_seqs 从 80 专用化到 1 后，缓存热启动快 1.21x；稳态 E2E 差异只有约 0.2%，不能宣称推理加速。
- 固定 KV cache 本身在缓存已热时只把初始化从 26.41 秒降到 25.60 秒（1.03x）。此前观察到的约 4x 必须归因于固定 cache、编译缓存命中和文件系统迁移的合成效果，不能单独记到固定 cache 名下。
- 冻结 vLLM baseline 的加权 E2E 是 936.15 ms，仍优于本轮最快的跨进程候选。截至第二轮没有发现可宣称的 strict-BF16 vLLM 之上加速，当时答案是 **1.00x（无可测提升）**；第五轮随后找到并验证了 `lm_head` 候选。

FP8 KV cache 仍是技术失败而非性能失败：它令 full-attention 后端切换到 FlashInfer，但 JIT 首先误用系统 CUDA 12.0 `nvcc`；改用虚拟环境 CUDA 13.3 `nvcc` 后，又与当前 CUDA 13.0 runtime headers 不兼容。因此没有生成可比较性能数据，也没有把该方向判成“算法无效”。考虑到 Qwen3.5-0.8B 只有 6 层 full attention、当前 batch=1 又主要受权重流限制，这个修复的预期全局收益很低，按预算停止继续追查。

### 推荐的快速复现实验配置

下面配置用于候选 screening，目标是缩短 agent 周转，而不是替代生产服务容量配置：

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_GDN_DECODE_KERNEL=cuda \
/home/aden/.venvs/qwen35-vllm-4060/bin/python \
  tools/benchmark_vllm_offline.py \
  --model /home/aden/models/Qwen3.5-0.8B \
  --output traces/<candidate>.json \
  --warmups 1 --trials 3 \
  --max-num-seqs 1 \
  --kv-cache-memory-bytes 536870912
```

只有候选越过噪声阈值后，才恢复生产容量设置并做交错 paired qualification。这样把“每次全量启动并深测”改成“缓存热身一次、短筛、最多两个晋级”，正面解决几十小时停留在实验测量的问题。

## 为什么严格 BF16 的 2 倍不可行

使用与活跃语言权重同量级的 BF16 read-only Triton stream，在本机得到：

- 输入读取 1,545,686,592 bytes，另有很小的每 program 输出；
- 中位 6.223 ms；
- 248.88 GB/s；
- 20 个原始样本范围为 6.199--6.317 ms。

排除视觉与未启用的 MTP 权重后，生成一个 token 至少消费 1,541,502,656 bytes 活跃权重。用实测 read service 作为乐观天花板：

`TPOT_floor = 1,541,502,656 / 248.878e9 = 6.194 ms`

该下界故意忽略 activation、GDN recurrent state、KV cache、同步、launch、采样和算术，所以只能偏乐观，不能把真实最优估得更慢。

| Workload | vLLM TPOT | 有效权重带宽 | 达实测读带宽比例 | 乐观最大加速 |
|---|---:|---:|---:|---:|
| prompt 128 | 6.497 ms | 237.26 GB/s | 95.3% | 1.049x |
| prompt 512 | 6.893 ms | 223.63 GB/s | 89.9% | 1.113x |
| prompt 2048 | 6.895 ms | 223.57 GB/s | 89.8% | 1.113x |

2 倍目标要求 3.25--3.45 ms/token，低于 6.194 ms 的乐观权重流下界。该结论不是“当前还没找到好 kernel”，而是“现有合同不允许移除造成下界的字节”。

## 对 agent 架构的修改

### 1. Intake 失败前置

`scripts/new_run.py` 原先只检查部分字段是否存在，会接受不符合仓库 JSON Schema 的 shape、额外字段和 objective enum，然后在下一阶段才失败。现在它在创建 run 目录前调用仓库的完整 Schema validator：

- operator、workload、hardware 任一不合法立即退出；
- 不创建半成品 run；
- 测试覆盖字符串 shape 这一真实失败形式。

这解决“实验合同从一开始就无效，但 agent 数小时后才发现”的浪费。

### 2. 目标可行性门

`optimizer_step.py` 现在识别 `models/feasibility_gate.json`：

- 校验 schema、目标、上界与 evidence SHA256；
- evidence 变化时返回 `BLOCK_INVALID_FEASIBILITY_GATE`，防止手改数字制造停止理由；
- 乐观资源下界仍排除目标时返回 `STOP_OR_REFRAME_INFEASIBLE_TARGET`；
- 给出必须重新选择的合同方向，不再要求 agent 为不可达目标凑候选、继续测量。

对应 opportunity-driven search 测试已通过。

### 3. 从局部利用率转成“可移除端到端时间”

真实 Transformers profile 显示一次 128-token prefill 的 ATen GPU 时间并非单一 GEMM 主导：

| ATen 类别 | GPU 时间占比 | 调用次数 |
|---|---:|---:|
| mm | 27.1% | 187 |
| copy | 22.7% | 4,149 |
| sum | 13.2% | 1,170 |
| mul | 11.0% | 1,692 |
| bmm | 10.1% | 235 |
| add | 5.6% | 1,481 |

因此候选生成的先验顺序应是：运行时/图捕获、GDN recurrence 融合、状态布局和物化消除、GEMM/epilogue；不应从一个 launch 参数 sweep 开始。到了成熟 vLLM 后，带宽闭合又会自动压低这些候选对 decode 的全局预期，避免把局部 1.8x 错写成模型 1.8x。

### 4. 候选执行路径证明

`candidate-smoke-result` 升级为 v3。smoke 除了正确性和目标值，还必须提供：

- `expected_path` 与实际观测的 `observed_path`，两者必须相等；
- 至少一个位于 run 内、SHA256 闭合的源码或执行证据；
- `FRESH`、`SOURCE_HASHED` 或 `NOT_COMPILED` 编译缓存策略。
- 运行时 `execution_proof`：kernel 实例数、插桩调用数或非编译直调 sentinel，并绑定到上述证据。对 `torch.compile`/CUDA Graph 候选禁止只用 host sentinel。

不满足时，`candidate_discovery.py` 把结果视为技术失败，不能进入 `QUALIFICATION_READY`；晋级凭证也会携带 reachability 记录。真实 vLLM harness 另增加 backend、source hash、空 cache root 三个启动前门和逐请求 GPU 遥测。源码/缓存门防止跑错版本，运行时计数则进一步防止 Python 条件在图捕获时被冻结、候选 custom op 实际没有进入 decode 图。

## 推荐的双赛道

### A. 严格等价赛道

保持当前 BF16 权重与单 token 自回归合同。目标应改为端到端 2%--8%，理论挑战上限约 10%。优先优化：

1. prompt 2048 的 GDN prefill 与 chunk/shape specialization，主要降低 TTFT；
2. 512/2048 decode 相对带宽流的约 10% residual，检查 recurrent/KV state 与 full-attention 额外流量；
3. 只在端到端预测超过测量噪声后实现新的 GDN fusion；
4. 最多两个 finalist 做完整 3×10 paired qualification。

### B. 2 倍目标赛道

它必须修改优化合同，但不应偷偷降低质量。建议依次研究：

1. W8A16/INT8 weight-only：权重字节理论减半，先测 perplexity、任务集与 token parity；
2. INT4/AWQ：带宽余量更大，但质量与 kernel 支持风险更高；
3. 可验证 speculative/MTP：用原模型验证接受 token，保持目标分布，但收益依赖接受率；
4. continuous batching：若真实 workload 允许并发，可把一次权重读取摊给多个请求，但它不是 batch=1 latency 的 2 倍。

这条赛道需要单独的 operator/workload contract、质量门和 baseline，不能与严格 BF16 数据混在一起领奖。

本轮还按 successive-halving 做了两个最小量化 smoke：

- 在线 FP8 成功使用 `CutlassFP8ScaledMMLinearKernel`，2-token parity 通过、模型显存降至 1.08 GiB；但 decode interval 从 BF16 的 26.18 ms 退化到 32.23 ms，热请求从 69.51 ms 退化到 94.82 ms，因此在完整 workload 前淘汰。
- `int8_per_channel_weight_only` preset 被解析为仅含 MoE spec，loader 明确报告 `Quantized 0 layers`。该实验在编译和计时前终止，避免把 BF16 重跑误报成 INT8 性能。

因此，“在线把现有 checkpoint 量化一下”并不能直接得到 2 倍。下一次有效的 2 倍尝试应使用明确量化了 dense linear 层的 checkpoint/配置，并先过相同的 2-token launched-mechanism gate。

## 第三轮：怎样在 4060 上真正快过 vLLM

这一轮把问题拆成三个不同合同：严格 BF16、可验证投机、量化运行时。新增 6 类自然请求（中文解释、Python 代码、算术推理、编辑、系统设计、翻译），每类固定生成 128 token，禁用 prefix/prompt cache，1 次 warmup、3 次 measurement，并轮换 case 顺序。

### 投机解码不是这个小模型的答案

合成 prompt 的输出高度周期性，n-gram-4 在该数据上得到 397.89 ms，对默认 vLLM 的 936.15 ms 看似有 2.353x；但换成自然请求后，默认 vLLM 为 940.98 ms，n-gram-4 变成 1813.21 ms（0.519x），且 6/6 输出都与默认路径不同。MTP-1 在自然请求上也只有 1138.91 ms（0.826x），仅 2/6 输出相同。llama.cpp 的 MTP-1 同样从 Q8 默认的 789.22 ms 退化到 1348.59 ms。

因此合成 n-gram 的 2.353x 是 benchmark exploitation，不是可推广的模型加速。agent 现在必须先通过代表性 workload 和输出合同，才能晋级候选。

### 轻量运行时与量化前沿

使用官方 llama.cpp Windows CUDA build `b10700`，将同一 BF16 checkpoint 转成 GGUF BF16，并另测官方 Q8_0、Q4_0。服务器保持常驻、单 slot、全部层在 GPU、Flash Attention 开启、提示缓存关闭。高性能状态下的量化 discovery 如下：

| 路径 | 加权 E2E | 输出速度 | 相对 vLLM BF16 | 数值合同 |
|---|---:|---:|---:|---|
| vLLM BF16 | 940.98 ms | 143.4 tok/s | 1.000x | 冻结 BF16 reference |
| llama.cpp Q8_0 | 789.22 ms | 175.4 tok/s | **1.192x** | 量化，需质量门 |
| llama.cpp Q4_0 | 700.18 ms | 201.7 tok/s | **1.344x** | 更激进量化，需质量门 |

后段低功耗状态下，llama.cpp BF16 为 2063.14 ms，而相邻时段 vLLM BF16 为 1222.63 ms，前者仍慢 1.687x，并且没有维持逐 token parity。这说明“只换掉 vLLM”并不会赢；实际胜点来自更少权重字节和适合 batch=1 的量化 kernel/轻量服务路径共同作用。

质量 discovery 使用本报告作为中英技术语料，512 context、8 chunks。llama.cpp BF16 perplexity 为 25.4609，Q8_0 为 25.4991（+0.15%），Q4_0 为 27.9038（+9.59%）。这不是下游任务 qualification，但足以把 Q8_0 排为当前质量优先候选，把 Q4_0 标为明确的延迟优先候选。

### 为什么现在还不能承诺稳定 1.34x

本机后段发生明显功耗状态漂移：vLLM 相同自然套件从 940.98 ms 变为 1222.63 ms（1.299x 变慢）；紧邻的 Q4_0 复测为 1147.39 ms，只领先 1.066x。独立的 21 点负载采样记录到中位 26.64 W、核心 780 MHz、显存 8001 MHz、GPU 利用率 87%，而设备默认功耗上限为 80 W。当前数据证明“存在胜出配置”，但还不是电源锁定、随机交错的 qualification。

下一道正式门应是：锁定笔记本性能模式；每个样本记录功耗、核心/显存时钟、温度；vLLM/Q8/Q4 随机交错；至少 3 warmups × 10 trials；再跑真实任务质量集。通过后才能把 1.19x 或 1.34x 写成产品承诺。

新增的可复现入口：

- `tools/benchmark_llamacpp_server.py`：启动持久 llama.cpp server 并跑自然请求套件；
- `tools/benchmark_llamacpp_perplexity.py`：量化质量 discovery；
- `tools/summarize_vllm_speculation.py`：识别合成投机假胜利；
- `tools/summarize_runtime_frontier.py`：合并速度、质量与功耗漂移证据；
- `models/vllm_speculation_search.json`、`models/runtime_frontier.json`：机器可读决策；
- `traces/llamacpp_server_natural_*.json`、`traces/llamacpp_quantization_ppl_c512_n8.json`：原始样本、输出与二进制/模型 SHA256。

## 第四轮：vLLM 是否已经最优，以及量化 vLLM 的实测

### “同参数量”不是同一推理合同

Q8、Q4 与 BF16 可以拥有相同数量的权重元素，但每个元素的表示和值都不同。对本机 batch=1 decode，BF16 每 token 至少读取约 1.542 GB 活跃权重；Q8/Q4 的主要收益来自减少这些字节，而不是找到了一个数学上等价、却凭空快数倍的 BF16 kernel。因此量化实现战胜 BF16 vLLM 不能证明其 runtime 更强，必须再与使用相同量化格式的 vLLM 比较。

vLLM 也不是抽象意义上的“理论最优”。但在本机高性能状态中，它的 6.497--6.895 ms TPOT 已经接近 6.194 ms 的乐观 BF16 权重流下界，对 batch=1 严格 BF16 只剩约 5%--11% 的理论空间。此时继续微调小算子不可能兑现 2x；要获得数量级更大的变化，必须减少权重字节、摊薄权重读取，或减少需要执行的目标模型 token step。

### GDN 局部候选：先撤回错误归因，再做可达性复验

Qwen3.5-0.8B 每 token 执行 18 个 GDN 层。packed recurrent Triton kernel 固定 `BV=32, num_warps=1, num_stages=3`。穷举 `BV={16,32,64,128}`、warps `{1,2,4,8}`、stages `{2,3,4}` 后，首轮曾因把 stock wrapper 与候选 direct launch 混测，误报 `num_stages=2` 快 1.315x；修正为两侧都 direct launch 后，原版为 53.51 us，局部最快 `BV=64, warps=1, stages=4` 为 38.99 us，表面快 1.372x，且输出和更新后的 FP32 recurrent state 均逐元素相等。

随后审计执行图发现：早先两组所谓“整模候选”运行时设置的是 `VLLM_GDN_DECODE_KERNEL=cuda`，实际走 C++ CUDA fused op，根本不会调用被修改的 Triton wrapper。因此此前把 4.44%/15.93% 变慢归因给两个 Triton launch 配置是错误的；这些数据只能证明跨进程功耗漂移，不能用于候选裁决，现正式撤回该归因。

加入 `--expect-gdn-decode-kernel` 可达性门后重新测试真实 Triton 路径：

| 可达路径 | 加权 E2E | 加权 TPOT | 输出 |
|---|---:|---:|---|
| Triton 原版 `BV=32, stages=3` | 893.92 ms | 6.789 ms | 6/6 exact |
| Triton 候选 `BV=64, stages=4` | 990.43 ms | 7.518 ms | 6/6 exact |

真实候选慢 10.8%，因此仍应淘汰，但现在淘汰理由来自正确执行路径。另一次同低频状态的 3-trial 对照中，CUDA 与 stock Triton 只差不足 1%，说明早先“CUDA 稳定快 5%”也不能跨功耗状态泛化。agent 必须同时记录实际 backend、候选源码 hash、编译缓存状态和 GPU 遥测，任何一项不满足都不得晋级或淘汰候选。

### vLLM W4A16/Marlin：速度通过，质量失败

下载并测试了第三方 `BlivionIaG/Qwen3.5-0.8B-AWQ-INT4` checkpoint。它是 compressed-tensors W4A16、group size 128，vLLM 成功选择 `MarlinLinearKernel`。在质量门加入前，其自然套件表面结果为 888.24 ms E2E、6.640 ms TPOT，相邻 BF16 原版为 1058.58 ms、7.959 ms，即表面约 1.19x。

但六个自然请求都退化为只有 2--3 个 distinct token 的特殊 token 循环，例如重复 `<think>\n\n</think>`。将 checkpoint 在 Transformers 中解压回 BF16 后仍得到同样循环，证明这是 checkpoint/量化结果失效，不是 vLLM Marlin 独有错误。该候选最终状态是 **FAIL / REJECT**，不能把 1.19x 当成可用结论。

benchmark 现新增低多样性退化门，并修复两类量化 checkpoint 兼容问题：

- 不再把“同一垃圾输出可以稳定重复”判作正确；
- 支持独立指定原模型 tokenizer/chat template，避免第三方量化目录缺模板；
- 权重证据改为 safetensors shard manifest hash，不再假设固定 shard 文件名；
- `compressed-tensors` 成为显式量化选项，报告会记录实际量化合同。
- 新增独立的 `scripts/audit_generation_quality.py`，让任何 runtime trace 都能在进入性能排行榜前先过低成本输出退化门；该门只排除明显坏结果，不替代 perplexity 与任务质量评测。

这个 checkpoint 的 734 MB 中仍保留 BF16 `lm_head`，而 248,320 x 1,024 的 tied vocabulary matrix 本身约 508 MB，并且每个生成 token 都要读取。其余层即使压到 INT4，也无法把全模型字节流缩成四分之一；再加上未量化层、GDN state、反量化与 launch 开销，正确实现的实际加速本来也会显著小于理论 4x。

当前结论是：量化 vLLM 在机制上应该参与公平竞赛，Marlin 已证明能执行并产生约 1.2x 的原始速度变化；但本次可获得的 AWQ checkpoint 质量失效，所以可用的 vLLM 量化冠军仍为空。下一步应从官方 BF16 权重生成分层量化前沿：先只量化 MLP，再加入 full-attention projection，最后才尝试 GDN q/k/v projection；每一级先过自然输出、perplexity/任务质量门，再测速度。

## 第五轮：严格 BF16 首个可复现胜出候选

### 为什么 stock vLLM 仍有缺口

vLLM 是面向多模型、多 GPU、多 batch 和高并发的通用 serving runtime，不保证每个 `M=1` GEMV 都拥有针对具体消费卡的最优 kernel。本模型的语言头是一个 `1 x 1024` 向量乘 `248320 x 1024` BF16 权重；在 Ada SM89 上，当前版本的 FlashInfer BF16 backend 只支持 SM100，CuTeDSL skinny GEMM 又只支持 SM90+，所以该形状最终退回 `torch.nn.functional.linear`/cuBLAS。

最初的顺序式单形状微基准把该投影测成 4076.13 us 对 2049.69 us（1.989x）；新增交错、每轮反转顺序的 paired 测量后，两边变为 2054.90 us 对 2046.79 us（1.004x）。这说明消费级笔记本 GPU 的升频/热状态足以让“先测完 torch、再测 Triton”的局部数字严重失真，微基准只能用于候选筛选，不能作为端到端收益的因果证明。Triton 核读取约 508 MB 权重，对应约 248 GB/s，仍接近本机实测显存读服务率；随机输入最大绝对差为 0.00390625、平均绝对差约 2.5e-7，argmax 相同。最终晋级依据是下面的整模复验，而不是 1.989x 的旧局部数字。

### 3×10 自然请求 qualification

只替换该 `lm_head`，不修改其余层、权重、精度、采样器或生成步数。candidate 与 stock 分别独立启动；每边 3 次 warmup、10 次测量，六类自然请求轮换顺序，每请求固定生成 128 token，并记录逐请求 GPU 功耗和时钟。

| 路径 | 加权 E2E | 加权 TPOT | 核心频率中位数 | 功耗中位数 |
|---|---:|---:|---:|---:|
| stock vLLM BF16 | 1262.67 ms | 9.623 ms | 930 MHz | 37.03 W |
| SM89 BF16 `lm_head` GEMV | 1070.37 ms | 8.128 ms | 915 MHz | 37.70 W |
| 加速 | **1.180x** | **1.184x** | 候选略低 | 近似相同 |

六类任务各自的 E2E 加速均在 1.173x--1.186x，TPOT 加速在 1.177x--1.193x；两边完整 128-token 序列 6/6 精确相等。候选没有靠更高核心频率取得收益。这是当前 4060 环境中首个通过同权重、同 BF16、自然 workload、逐 token 回归和功耗遥测的 vLLM 之上候选。

为排除“改源码导致另一份 Inductor 图”这一混杂因素，又把补丁改成默认关闭、由 `VLLM_SM89_BF16_LM_HEAD=1` 开启；stock 与 candidate 因而共享完全相同的 `utils.py` SHA256（`a2b0d1ac...3dc23`）和编译缓存。按 C1-S1-S2-C2 顺序各做 1 warmup × 3 trial：

| 同源复验 | 加权 E2E | 加权 TPOT | 核心频率中位数 | 功耗中位数 |
|---|---:|---:|---:|---:|
| candidate C1 | 1060.27 ms | 8.084 ms | 915 MHz | 37.28 W |
| stock S1 | 1262.98 ms | 9.683 ms | 952.5 MHz | 36.69 W |
| stock S2 | 1264.43 ms | 9.684 ms | 922.5 MHz | 36.71 W |
| candidate C2 | 1058.61 ms | 8.075 ms | 892.5 MHz | 37.61 W |
| 两边均值之比 | **1.193x** | **1.199x** | 候选更低 | 近似相同 |

两组配对比较都为 6/6 token-exact，且每个任务均胜出。paired 微基准与整模结果不矛盾于“候选有效”的判定：前者证明局部计时会受前序 workload 影响，后者才覆盖真实权重布局、调用节奏、CUDA Graph 与 runtime 调度。

随后在用户目录安装 Nsight Systems 2025.5.1，并用 `--cuda-graph-trace=node` 成功展开 CUDA Graph。六个 32-token 自然请求共生成 192 token，其中 186 个稳定 decode 图步骤。`lm_head` 候选被观察到 192 次，平均 2057.54 us；主干 cuBLAS GEMV 被观察到 21,204 次，即每个 decode 步骤 114 次，总计约 5.036 ms/step。这个 timeline 同时给出了后续搜索的全局预算，但 node tracing 会扰动短 kernel，因此只用于机会排序和执行路径证明，不把它的百分比当作生产 qualification。

这个结论的边界也很明确：它是单请求延迟胜出，不是高并发吞吐 SOTA；逐 token 相等覆盖当前六类 qualification 请求，不等于对所有可能输入证明浮点 bitwise 等价；历史高功耗状态与当前低功耗状态不能横向混排。补丁位于 `patches/vllm_0.28.1_sm89_bf16_lm_head.patch`，机器可读证据位于 `models/sm89_lm_head_candidate.json`。

### 为什么没有把同一 GEMV 铺满主干

孤立微基准曾预测多个主干投影也会变快。旧实验即使使用了新 `VLLM_CACHE_ROOT`，其 `x.numel()==x.shape[-1]` Python 分支仍可能在 `torch.compile` 动态图捕获时被冻结；因此旧的 GDN/MLP/全开消融没有证明候选 custom op 被执行。此前写下的“慢 1.0%--2.4%”因果归因现正式撤回，原始数据只保留为不可达实验记录。

修复后的候选让所选权重形状无条件经过 opaque custom op，并在 op 内部决定 `M=1` 使用 Triton、其他形状回退 `F.linear`。Nsight 分别观察到预期的 **3,540/3,540** 和 **12,468/12,468** 个候选 kernel，证明它们真正进入了 decode 图：

| 可达候选 | 局部证据 | 非 profiler 端到端筛选 | 输出合同 | 决定 |
|---|---:|---:|---:|---|
| GDN `8192x1024` | 80.46 → 72.54 us | profiler E2E 1.022x | 5/6 exact | 淘汰 |
| GDN + MLP gate-up | kernel 均下降 | C-S-C 平均 TPOT 1.016x | 2/6 exact | 淘汰 |
| attention stacked QKV `5120x1024` | micro 1.625x | screening TPOT 1.007x | 2/6 exact | 淘汰 |

这里还修正了 attention QKV 的真实输出宽度：vLLM 把 `q=4096, k=512, v=512` 堆叠为 5120，而不是旧记录中的 3072。结果说明这些局部替换确实能减少 kernel 时间，但独立替换每个 GEMV 的全局收益只剩约 0.7%--1.7%，且改变 BF16 累加顺序后未通过当前 token-exact 合同。它们不能进入补丁；下一代候选必须通过跨投影融合、持久化或物化消除，移除流量/launch，而不是继续扫单个 GEMV 参数。

### 下一轮由全局机会图决定

Nsight 结果已通过公共 `kernel_opt.py opportunity` 入口写回正式机会图，并覆盖 12 个不同 rewrite family。按“预期全局收益 × 置信度 / 实现分钟数”排序如下：

| 排名 | 机会 | 可能移除的时间/step | 实现预算 | 含义 |
|---:|---|---:|---:|---|
| 1 | normalization/epilogue 融合 | 20--80 us | 60 min | 小而较便宜 |
| 2 | recurrent state 融合与布局 | 30--100 us | 90 min | 需跨算子边界 |
| 3 | decode graph 小 kernel 压缩 | 20--100 us | 60 min | node tracing 下低置信度 |
| 4 | 分段持久投影调度 | 20--120 us | 90 min | 简单 QKVZ+BA 拼接已被快速否决，降为低置信度 |
| 5 | attention/KV 布局协同 | 10--50 us | 90 min | 预期收益较小 |
| 6 | 继续微调 lm-head | 0--50 us | 45 min | 已接近带宽屋顶，停止无界 sweep |

因此 agent 下一步不能再随机挑一个 launch 参数，而要先验证真实数据流，再按收益/成本比选择架构。这里的区间是经验搜索先验，不是理论最优证明；实际结果必须再由运行时可达性、正确性和交错 A/B 更新。

### 第八轮：先证明融合合法，再用廉价整模型筛选快速止损

源码审计表明，vLLM 已经合并了 MLP `gate+up`、全注意力 `Q/K/V/gate` 和 GDN `Q/K/V/Z`；跨越 SiLU、attention、recurrent state 或 residual/RMSNorm 的“跨投影融合”存在真实数学依赖，不能通过调度直接消掉。唯一尚未合并、又共享相同输入的重复边界，是每个 GDN 层的 `8192x1024 QKVZ` 与 `32x1024 BA`，24 层中出现 18 次。机器可读的数据流审计见 `models/qwen35_projection_dataflow_audit.json`。

实现的一次性 `8224x1024` 合并投影成功加载 checkpoint，并在 6 个自然请求、每请求 16 token 的廉价筛选中保持 6/6 逐 token 相等；但相对保留专用 lm-head 的 stock 路径，TPOT 仅为 **0.943x**、E2E 仅为 **0.931x**，即反而慢约 5.7% 和 6.9%。最可能的原因是把 tile 友好的 8192 行主投影改成 8224 行后，主 GEMV 调度退化超过省掉 32 行 BA launch 的收益；这项原因尚未由 kernel counter 证明，所以只作为解释性推断。候选源码和原始结果保存在 `candidates/gdn-qkvz-ba-fusion/`，当前结论是 discovery screen-out，不是 production qualification。

这个失败直接校准了搜索图：原排名第一的“跨投影融合”从 200--500 us、高置信度、120 分钟预算，降为只剩分段 persistent schedule 的 20--120 us、低置信度、90 分钟预算，排名降到第四。更重要的是，agent 现在不会因为“2x 目标数学上不可达”就原地停止：`residual_search_policy` 可以在冻结合同内授权有收益下限和总时限的 best-feasible 搜索；每个新候选还必须提交 hash-bound `dependency_contract`，先证明 DAG 与数值顺序合法，才允许注册和编译。

这一轮也修正了原先过强的理论推断：整模型权重流下界能排除严格 BF16 的普遍 2x，但不能证明 stock vLLM 的每个子算子已高效。理论模型应输出“剩余总预算”和“按算子可移除时间”两个层级；只要某个大算子明显低于同机带宽屋顶，局部专用化仍可能兑现两位数的端到端收益。

### 第九轮：用接受率经济账淘汰 MTP，而不是只看最终延迟

Qwen3.5-0.8B checkpoint 自带一层 MTP 权重。一个看似有潜力的架构方向是让 target 一次验证两个位置，把一次主模型权重读取摊给多个 token。先做的 `M=2, N=248320, K=1024` BF16 LM-head 微基准确实证明 cuBLAS 已经能共享这次权重读取：交错测量中 cuBLAS 为 2085.27 us，专用 Triton 为 2057.80 us，只快 **1.013x**。这远低于 3% 的晋级阈值，因此没有把 M=2 kernel 接进生产路径。

随后在保留现有 SM89 M=1 LM-head 优化的前提下，对 MTP-1 做了相邻自然请求筛选，并开启 vLLM 的逐请求接受率统计。64-token 六用例结果为：

- 非投机路径加权 TPOT 7.907 ms，MTP-1 为 8.859 ms，即 **0.893x**；E2E 为 **0.867x**。
- 216 个 draft 中接受 164 个，draft 接受率 **75.93%**，每轮平均产出 1.759 个 token。
- 按实测 `cycle_cost = speculative_TPOT × mean_acceptance_length` 反推，一轮 proposer+verify 约 15.585 ms；要打平非投机路径，MTP-1 接受率需约 **97.11%**。
- 即使假设 100% 接受且 cycle cost 不变，TPOT 乐观下界也只有 7.793 ms，相对 control 的上限仅 **1.0146x**，仍低于 3% 晋级门槛。
- 64-token 输出仅 2/6 与非投机路径逐 token 相同。投机验证在概率语义上不应改变 target 分布；这里更可能是 M=1 Triton 与 M=2 cuBLAS 的归约/舍入路径不同，在 greedy 临界 logits 上放大成 token 分叉。因此它也不满足本 run 更严格的逐 token 冻结合同。

这次没有继续深挖 MTP 内部小 kernel，因为“完美接受上限”已经给出止损证明。新增通用 `scripts/analyze_speculation_economics.py`，会把接受率、每轮成本、打平所需接受率、完美接受上限与 token 一致性一起写入决策；`tools/benchmark_vllm_offline.py` 在启用 MTP/ngram 时也会自动记录 per-request acceptance metrics。由此避免两类典型误判：只因合成重复文本接受率高就宣称 2x，以及在理论上最多只剩约 1% 时继续花数小时调 proposer。

机器可读证据：

- `microbench_candidates/bf16_triton_lm_head_m2_sm89.json`
- `traces/vllm_natural_sm89_lmhead_mtp_acceptance_control_w1_n1_t64.json`
- `traces/vllm_natural_sm89_lmhead_mtp1_acceptance_w1_n1_t64.json`
- `models/sm89_mtp1_acceptance_economics.json`

### 第十轮：修复热 L2 微基准造成的“假大收益”

进一步扩展 LM-head 搜索到 72 组 block/warp/stage 调度后，样本最优配置相对当前配置的交错复测只有 **0.9985x**，因此保持当前 `BLOCK_N=4, num_warps=8, num_stages=1`。Nsight 中后续 argmax 平均只有 19.72 us、占 GPU 时间约 0.235%；即使完全融合也达不到当前物质性门槛，所以没有为 greedy-only 特例破坏通用 logits 接口。

更重要的发现来自 backbone GEMV：本机通过 `torch.cuda.get_device_properties` 读到 32 MiB L2，而单个 GDN QKVZ、MLP gate/up、MLP down 权重分别只有 16 MiB、14 MiB、7 MiB。原微基准连续重复同一个矩阵，权重可以驻留 L2；真实 decode 每 token 依次读取约 1.542 GB 活跃权重，不可能保持这些矩阵驻留。因此原先约 1.6x 的 isolated GEMV 数字系统性高估生产收益。

微基准现在增加 64 MiB 驱逐缓冲：每次计时前先读写该缓冲，CUDA event 放在驱逐之后，因此只测一个冷缓存 GEMV。相同候选的结果变成：

| 投影 | 热缓存速度比 | 冷缓存速度比 | 每 token 次数 |
|---|---:|---:|---:|
| GDN QKVZ | 1.629x | 1.041x | 18 |
| MLP gate/up | 1.571x | 1.071x | 24 |
| MLP down | 0.965x | 1.019x | 24 |

这解释了为什么此前热微基准预测能省接近 1.8 ms，而 Nsight/整模型只兑现约 0.13--0.25 ms。为了检查是否仍有安全的小胜利，又实现并运行了只替换 `mlp_down` 的生产候选：直接 M=1/M=2 数值探针均与 torch 完全相等，但自然请求中随着自回归误差传播仍只有 5/6 逐 token 相同；相邻全模型筛选的 TPOT 为 7.9764 -> 7.9689 ms（**1.00094x**），E2E 为 **0.9982x**。候选低于 1% 晋级线并在 E2E 上回退，已经淘汰，安装环境恢复到 hash `a2b0d1...` 的 LM-head-only 源码。

agent 因此新增一条强制规则：如果孤立权重小于 L2、但生产迭代的唯一权重流大于 L2，则只能用冷缓存单次计时或整模型 trace 排名；热重复结果不能支持晋级。这不是单纯增加一个 profiler，而是把“局部实验的 cache 初始条件是否与全局执行一致”加入因果合同，直接解决候选在错误局部指标上反复迭代的问题。机器可读审计见 `models/sm89_cache_residency_gate.json`。

复现补丁与 qualification：

```bash
cd /home/aden/.venvs/qwen35-vllm-4060/lib/python3.12/site-packages
patch -p1 < /mnt/d/codes/kernel_opt_agent/runs/20260902_qwen35_08b_e2e_sm89_v1/patches/vllm_0.28.1_sm89_bf16_lm_head.patch

cd /mnt/d/codes/kernel_opt_agent/runs/20260902_qwen35_08b_e2e_sm89_v1
VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_GDN_DECODE_KERNEL=cuda \
VLLM_SM89_BF16_LM_HEAD=1 \
/home/aden/.venvs/qwen35-vllm-4060/bin/python tools/benchmark_vllm_offline.py \
  --model /home/aden/models/Qwen3.5-0.8B \
  --output traces/reproduction.json \
  --prompt-suite natural --new-tokens 128 \
  --warmups 3 --trials 10 --max-num-seqs 1 \
  --kv-cache-memory-bytes 536870912 \
  --expect-gdn-decode-kernel cuda --expect-sm89-lm-head triton --gpu-telemetry \
  --expect-source-sha256 /home/aden/.venvs/qwen35-vllm-4060/lib/python3.12/site-packages/vllm/model_executor/layers/utils.py=a2b0d1ac0600564dae318afc544d7876e13b2e847e44cb1d5632bec7d213dc23
```

正式比较 stock 时使用同一份已打补丁源码，取消 `VLLM_SM89_BF16_LM_HEAD`，并指定 `--expect-sm89-lm-head stock`；这样无需通过反向打补丁制造两份不同源码。若修改的是编译图内部路径，还应给每个候选设置独立、初始为空的 `VLLM_CACHE_ROOT` 并增加 `--require-empty-vllm-cache-root`；本候选的 `lm_head` 位于已编译 backbone 之外，但仍保留 source hash 与实际路径门以防跑错版本。

### 第十一轮：不量化，直接在 vLLM 内继续压缩 BF16 decode

这一轮专门回答“vLLM 是否已经最优、量化是否必然更快”。结论是否定的：vLLM 优化的是多模型、多 GPU、多 batch/并发与通用接口的生产折中，不是 RTX 4060 Laptop、Qwen3.5-0.8B、batch=1、固定 decode shape 的理论最优。量化能减少权重流量，但也增加反量化、类型转换、量化 kernel 覆盖与质量校准成本；当 LM-head、kernel launch、采样或状态更新占主导时，量化并不自动兑现等比例端到端收益。

先尝试复用 vLLM 已存在的 MTP fused GDN post-convolution+normalization kernel处理普通单 token decode。候选可达且 6/6 token 完全一致，但 E2E 只有 **0.968x**、TPOT **0.996x**；MTP kernel 的固定工作抵消了少一次 normalization launch 的收益，立即淘汰。这是 opportunity map 中 normalization/epilogue 方向的有证据止损，不再继续扫参数。

随后实现了分段 GDN 投影：QKVZ `8192x1024` 和 BA `32x1024` 仍使用各自适合的 tile，但由一个 Triton launch 分派两个 segment，避免上一轮粗暴拼成 8224 行破坏主投影调度。结果分三层验证：

| 证据层 | stock | candidate | 结果 |
|---|---:|---:|---:|
| 热缓存孤立投影 | 91.135 us | 51.343 us | **1.775x** |
| 64 MiB 驱逐后的冷缓存投影 | 111.616 us | 99.328 us | **1.124x** |
| lm-head 已开启的整模型 C-S-S-C | TPOT 8.280 ms | TPOT 7.961 ms | **1.040x** |

Nsight Systems 2025.5.1 在 186 个 decode step 中观察到候选 kernel **3348/3348** 次，恰好等于 18 个 GDN 层乘 186；每 step 的投影调用由 36 次降为 18 次，cuBLAS 总调用由 114 次降为 78 次。说明收益来自真正进入生产图的 launch/调度变化，而不是不可达代码或只在微基准成立。

最后使用相同 vLLM 源码和 BF16 模型做直接 C-S-S-C：对照关闭两个专用开关，候选同时开启 SM89 lm-head 与 segmented GDN；首个 control/candidate 各用独立空编译缓存，六个自然请求、每请求 64 token、每进程 1 次 warmup + 3 次测量。

| 前沿 | 加权 E2E | 加权 TPOT | 约合 decode tok/s | 相对原版 BF16 vLLM | 跨模式 token 一致性 |
|---|---:|---:|---:|---:|---:|
| 原版 vLLM 均值 | 653.063 ms | 9.750 ms | 102.6 | 1.000x | 基准 |
| 组合 BF16 均值 | 548.625 ms | 8.125 ms | 123.1 | **1.190x E2E / 1.200x TPOT** | 3/6 |
| 严格 BF16 前沿（先前 128-token qualification） | 见 `models/sm89_lm_head_abba.json` | 见同左 | — | **1.193x E2E / 1.199x TPOT** | **6/6** |

两次 control 彼此 6/6 相等，两次组合候选也彼此 6/6 相等；组合路径是确定性的。模式之间只有 3/6 完全相同，原因是 segmented Triton 与 cuBLAS 的 BF16 归约顺序不同，小数值差异在 greedy 临界 logits 上经自回归放大。因此这里明确保留两条前沿：

- **严格前沿**仍是 lm-head-only：对 stock 6/6 token-exact，已有 128-token、同源 C-S-S-C 的约 1.20x 证据。
- **数值兼容 discovery 前沿**是 lm-head + segmented GDN：本次直接 BF16 对照同样约 1.20x，并证明 GDN 局部还有约 4%，但在更大任务质量评测前不能宣传为生产 winner，更不能把 BF16 标签当作 bitwise-equivalent。

只替换后半 GDN 层的折中也已测试：E2E **0.994x**、TPOT **1.003x**，跨模式仍为 3/6 exact，因此同时失去物质性收益与严格复现价值，已经淘汰。这个结果阻止 agent 在“选哪些层”上继续无界组合搜索。

复现激进候选时，先应用已有 lm-head 补丁，再应用：

```bash
cd /home/aden/.venvs/qwen35-vllm-4060/lib/python3.12/site-packages
git apply --ignore-space-change /mnt/d/codes/kernel_opt_agent/runs/20260902_qwen35_08b_e2e_sm89_v1/candidates/gdn-segmented-projection/vllm_qwen_gdn_segmented_projection.patch

VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_GDN_DECODE_KERNEL=cuda \
VLLM_SM89_BF16_LM_HEAD=1 \
VLLM_SM89_SEGMENTED_GDN_PROJECTION=all \
VLLM_CACHE_ROOT=/tmp/vllm-segmented-gdn-fresh \
/home/aden/.venvs/qwen35-vllm-4060/bin/python /mnt/d/codes/kernel_opt_agent/runs/20260902_qwen35_08b_e2e_sm89_v1/tools/benchmark_vllm_offline.py \
  --model /home/aden/models/Qwen3.5-0.8B \
  --output traces/reproduction_combined.json \
  --prompt-suite natural --new-tokens 64 --warmups 1 --trials 3 \
  --max-num-seqs 1 --kv-cache-memory-bytes 536870912 \
  --expect-gdn-decode-kernel cuda --expect-sm89-lm-head triton \
  --require-empty-vllm-cache-root --gpu-telemetry
```

机器可读总表为 `models/sm89_combined_bf16_frontier.json`，候选机制、冷/热缓存、Nsight 与淘汰实验为 `candidates/gdn-segmented-projection/summary.json`。这轮证明的是“同精度通用框架仍有约 20% 专用化空间”，不是理论全局最优；要继续接近最优，应优先做质量门和更低层的持久化/图边界改写，而不是默认转向量化或继续扫无关 launch 参数。

### 第十二轮：质量门、剩余下界与 MTP 架构止损

上一轮组合候选只有逐 token 一致性，无法回答“输出变了但能力是否退化”。本轮从 OpenAI 官方 `grade-school-math` 仓库固定 commit `3101c7d5072418e28b9008a6636bde82a006892c`，用相同 BF16 权重、greedy、关闭 thinking、固定随机子集做 512 题 GSM8K 配对筛选：

| 路径 | 正确题数 | 正确率 | 相对 control |
|---|---:|---:|---:|
| lm-head-only control | 47/512 | 9.180% | — |
| lm-head + segmented GDN | 46/512 | 8.984% | -0.195 个百分点 |

两边最终数值答案一致 485/512，完整 token 一致 477/512；配对结果为双方都对 45、仅 control 对 2、仅 candidate 对 1、双方都错 464，McNemar 双侧精确检验 `p=1.0`。因此没有检测到候选导致的任务正确率退化，可以把它从 `QUALITY_UNKNOWN` 升为 **limited task-quality pass**；但 512 道低正确率数学题不能证明全任务、采样生成或生产质量等价，严格前沿仍保持 lm-head-only。

同时把“还可能快多少”分成三个边界：

- 当前严格候选 TPOT 为 8.079 ms；全部活跃 BF16 权重按本机校准读取服务计算的乐观地板是 6.194 ms。即便假设 attention、状态更新、同步和采样全部免费，绝对乐观上限也只有 **1.304x**，所以它只能用于排除 2x，不能冒充可达到的预测。
- `lm_head` 每步必须读取 508,559,360 bytes，带宽地板 2.0434 ms，实测候选 2.0575 ms，已经达到校准服务屋顶的 **99.31%**；72 个后续 schedule 的最好配对结果也只有 0.9985x。继续调 block/warp 是已关闭死路。
- 除 `lm_head` 外的 backbone 权重地板约 4.150 ms，而 profile 中 cuBLAS projection service 为 5.036 ms；原始差额 0.885 ms 仍包含不可消除的计算，因此下一次搜索必须改变数据移动、跨层持久化或状态布局，不能把整段差额都当成收益。

为了验证“让 MTP 一轮处理两个位置，是否能绕过单 token 权重流地板”，本轮还把所有主要 `M=2` projection 做了 64 MiB 冷缓存微基准。若乐观地把每个 shape 都替换成各自最佳 Triton schedule，推算每个 speculative cycle 可省 **1.462 ms**；但按实测平均接受长度 1.759，MTP 的预测 TPOT 仍是 8.028 ms，相对非投机路径只有 **0.985x**。达到 1.03x 晋级门还缺独立的 **0.618 ms/cycle**。因此没有花时间把这批很漂亮的局部 kernel 接入生产图：局部加权约 1.20x--1.27x，不等于端到端能赢。

这形成了 agent 应采用的决策方式：同时保留严格 token-exact 前沿和有限任务质量前沿；先用整步经济账判断架构方向，再实现 kernel；每个方向必须有收益上限、晋级阈值和退出证书。机器可读结论在 `models/sm89_strict_bf16_residual_certificate.json` 与 `models/sm89_mtp_m2_projection_bound.json`。当前可以证明的是“测试过的有界候选集中，lm-head-only 是严格最优”；不能证明它是所有未知架构中的理论最优。

### 第十三轮：让止损真正进入调度器，并纠正 recurrent 的 dtype 假象

继续运行 agent 后发现一个架构缺口：报告和 opportunity observation 虽然写了“已拒绝”“已到屋顶”，但机会状态机只认识 `UNIMPLEMENTED / IMPLEMENTING / OBSERVED / HAS_SURVIVOR`。自由文本不会阻止下一轮重新选择同一机会，因此 agent 仍可能回到已经扫过 72 个 schedule 的 `lm_head`，这正是长时间卡死的机制性原因。

现在新增正式的 `CLOSED` 生命周期与 `opportunity close/reopen`：

- 关闭必须记录 disposition、全局止损理由、run 内证据 SHA-256 和至少一个明确的重开条件；
- `CLOSED` 的排序分数固定为 0，并从方法匹配、候选注册、候选续跑和 next-action 路由中排除；
- 所有机会关闭时，调度器返回 `OPPORTUNITY_PORTFOLIO_CLOSED`，不会自行制造新测量；
- 证据被改写会使机会图校验失败；只有显式 `reopen` 并记录发生变化的条件，预算才会恢复。

当前 run 已把 `lm_head` residual 关闭：它绑定 `models/sm89_strict_bf16_residual_certificate.json`，重开条件仅包括 shape 改变、设备带宽服务曲线实质变化，或出现能消除完整 logits 物化的新算法。真实执行 `kernel_opt.py next` 已跳过它。

recurrent 方向还修正了一个重要实验口径。旧孤立 sweep 使用 FP32 state，得到 `BV64/W1/S4` 约 1.37x；但本模型 `mamba_ssm_cache_dtype=auto` 在 BF16 模型上实际使用 BF16 state。生产 dtype 下的有限复测结果为：

| recurrent screen | stock | 最好候选 | 结论 |
|---|---:|---:|---:|
| BF16 state 热循环 | 32.171 us | 30.290 us | 1.062x；投影到 18 层仅省 33.9 us/step |
| 64 MiB 驱逐、交替单 launch | 15.360 us | 20.480 us | **0.750x** |
| 真实可达整模型旧候选 | — | — | **0.903x**，6/6 token-exact |

另外只补测了 18 个 `BV64/BV128 × 2/4/8 warps × 2/3/4 stages` ownership 点。它们的当前输出都相同，但 recurrent state 全部出现最多 `2.98e-8` 的差异，因此在严格状态转移合同下计时前就被淘汰。结论不是关闭整个 recurrent 架构方向，而是关闭 `BV/warp/stage` 扫参和复用通用 MTP fused kernel：下一候选必须改变 convolution→recurrence→gated norm 的 producer-consumer ownership 或 state layout，并先预测至少 50 us 的整步收益。

重新标定后，normalization 的 likely interval 从 20--80 us 下调为 0--30 us；recurrent-state fusion 成为最高有效机会。机器可读证据在 `models/sm89_recurrent_state_search_bound.json`。这一轮没有产生新的端到端 winner，但消除了一个错误的 1.37x 先验，并保证 agent 不会因它再次浪费数十小时。

### 第十四轮：融合不是天然更快，必须按 vLLM 的 CUDA Graph 口径判定

针对当前最高优先级 recurrent 机会，实现了一个真正改变 ownership 的候选：stock 每层用 64 个小程序更新状态，再单独 launch gated RMSNorm；候选改为每个 value head 一个程序，顺序处理四个 BV32 state slice，并在同一程序完成 128 维 gated RMSNorm。候选的 recurrent state 与最终 BF16 输出均逐位一致，最大差异都是 0。

如果按普通 Python 连续提交计时，候选看起来从 100.504 us 降到 23.259 us、达到 4.321x；但这个数字混入了 stock `rmsnorm_fn` 的张量分配和两次 host submission，不能代表 vLLM 的真实 decode。按 vLLM 生产路径的 CUDA Graph replay 重新测量后，结论反转：

| 口径 | stock recurrent + norm | 融合候选 | 相对 stock |
|---|---:|---:|---:|
| CUDA Graph replay | 16.398 us | 21.967 us | **0.747x** |
| 64 MiB 驱逐、交替单次 | 21.504 us | 24.576 us | **0.875x** |

原因不是融合数学错误，而是并行度：stock 的四个 BV32 tile 可并行，候选把它们串行到 16 个 head program 中；省掉一次 launch 和重复 q/k 读取不足以补偿。该设计已拒绝，不接入整模型。recurrent 机会仍未整体关闭，但重开候选必须同时满足“保留四 tile 并行度”和“提供严格 128 维归一化同步边界”，并先在 CUDA Graph 下胜出。机器可读退出证书为 `models/sm89_recurrent_norm_fusion_stop.json`。

这个实验也回答了为什么“自己写个 fused kernel”经常打不过 vLLM：错误计时口径能制造数倍假收益，而 vLLM 的 CUDA Graph 已把 host launch 开销压低；真正竞争的是 GPU 内部 occupancy、访存与依赖链，不是源码里 kernel 数量的多少。

### 第十五轮：从 vLLM 热点继续融合，局部 1.70x 只转化为整模 1.014x

recurrent 方向又实现了一个四 warp/head 的 CUDA 版本，用 cooperative groups 在 128 维归一化前同步，试图同时保留四个 BV32 tile 的并行度。它的即时 BF16 输出与 stock 相同，但 recurrent state 有 10 个元素不同、最大差异 0.00390625；更关键的是 CUDA Graph replay 从 16.009 us 退化到 34.718 us（0.461x），64 MiB 驱逐口径也只有 0.395x。因此 `sm89-recurrent-state-fusion` 已用闭环证书正式关闭，而不是继续围绕 warp、tile 和 stage 做无穷扫参。机器可读证据为 `models/sm89_recurrent_opportunity_closure.json` 和 `microbench_candidates/gdn_recurrent_norm_cuda_sm89.json`。

调度器随后把完整 decode CUDA Graph 的短 kernel 尾部排到第一位。本轮选择 24 个 MLP 层都存在的 producer-consumer 边界，把 `SwiGLU = silu(gate) * up` 融入 BF16 down projection。孤立 CUDA Graph 微基准从 45.169 us 降到 26.522 us，达到 **1.703x**，按 24 层简单求和似乎每 token 可省 0.448 ms。候选随后通过 direct custom op 接入 vLLM，并在两次独立冷缓存编译的 CUDA Graph capture 中留下可达日志，排除了“写了代码但图里没有走到”的假阳性。

严格的 control-candidate-candidate-control、6 个自然提示、每个进程每例 3 次、固定 64 token 结果如下：

| 指标 | lm-head-only control | fused SwiGLU-down | 相对 control |
|---|---:|---:|---:|
| 平均 E2E | 549.157 ms | 540.467 ms | **1.016x** |
| 平均 TPOT | 8.0652 ms | 7.9515 ms | **1.014x** |
| 平均 TTFT | 40.066 ms | 40.365 ms | 0.993x |
| 64-token 逐 token 一致 | 6/6 | 4/6 | 未通过严格门 |

候选两次运行彼此逐 token 一致，对照两次也彼此一致；分叉只发生在两种实现之间，首个差异分别位于中文解释的第 22 个 token、翻译的第 34 个 token。这说明它不是测量噪声，而是 GEMV reduction order 改变造成的数值路径变化。实际每 token 只省 0.114 ms，局部求和预测向整模转化约 **25%**：CUDA Graph 内存在重叠、调度和非目标开销，不能把各层微基准收益直接相加。

因此这个候选记录为 **MEASURED_REJECT**：它确实比 vLLM 对应局部路径快，也让完整模型稳定快约 1.4%，但同时低于 1.03x 晋级门并违反严格 token-equivalence，继续围绕该实现调 block/warp 的期望价值很低。这个负结果反而把 agent 的搜索原则变得更清楚：先验证图中可达性，再做同源整模筛选；局部大胜若无法转化，立即生成退出证书并转向其他高占比边界。机器可读汇总为 `candidates/fused-swiglu-down/summary.json`。

### 第十六轮：修复“候选根本没进入图”的假实验，并关闭独立 backbone GEMV 扫参

回看早期 backbone GEMV 实验时发现一个关键可达性漏洞：代码在 `default_unquantized_gemm` 的 Python 层用 `x.numel() == x.shape[-1]` 判断是否为单 token decode；但 torch.compile 首先在 prefill 形状上捕获动态图，这个分支被提前专门化为 false。换源码、清空缓存甚至端到端跑通，都不能证明候选真的进入 decode CUDA Graph。旧的 shape-guard-only 结论因此继续保留原始数据，但标记为 `RETRACTED_UNREACHABLE`，不再用于性能归因。

新的实现把 shape 选择放在编译期稳定的 weight 维度上，把 decode/prefill 判断移入一个 runtime custom-op：单 token 时调用 SM89 Triton GEMV，其余形状回退 `F.linear`。在两次独立冷缓存编译中，CUDA Graph capture 明确记录 `N=7168,K=1024` 和 `N=1024,K=3584` 候选被捕获，解决了“写了算子但完整模型没有执行”的验证盲点。

严格的 control-candidate-candidate-control 结果表明，替换 MLP gate/up 与 down 两类 projection 后，局部收益这次确实转化到了完整模型：

| 指标 | lm-head-only control | reachable MLP GEMV | 相对 control |
|---|---:|---:|---:|
| 平均 E2E | 552.719 ms | 526.848 ms | **1.049x** |
| 平均 TPOT | 8.1543 ms | 7.7175 ms | **1.0566x** |
| 候选重复一致 | 6/6 | 6/6 | 确定性通过 |
| 跨实现逐 token 一致 | 6/6 | 2/6 | 严格门失败 |

以原始 stock vLLM 的约 9.684 ms TPOT 为参照，这条 BF16 discovery 路径累计约 **1.25x**；但它不能替代严格前沿。512 题 GSM8K 配对筛选中，control 为 47/512、候选为 44/512，McNemar 双侧精确检验 `p=0.25`，尚未检测到统计显著退化，但 3 个不一致的正确样本全部只由 control 答对，方向性信号不适合生产晋级。

分 shape 消融把问题进一步定位到 reduction order：gate/up 单独替换约 1.047x、只有 2/6 exact；down 单独替换约 1.036x、达到 5/6 exact。为了检验“只改变归约树能否保住收益并恢复输出”，又对 down projection 做了有界的 2/4/8-warps 实验：

| down schedule | TPOT 加速 | 跨实现 token exact | 决策 |
|---|---:|---:|---|
| 2 warps | 1.0216x | 3/6 | 更慢且不更准，拒绝 |
| 4 warps | **1.0361x** | **5/6** | 最好候选，但严格门失败 |
| 8 warps | 1.0185x | 3/6 | 更慢且不更准，拒绝 |

因此这次没有把 4-warp 的 5/6 当成“差不多正确”，也没有继续无界扫 launch 参数。独立 backbone GEMV substitution 已按 `MEASURED_REJECT_STRICT` 结案；只有出现能保留 stock reduction path，或能删除 producer-consumer materialization/权重读取且预测整步收益至少 3% 的新算法时才重开。安装环境已恢复到哈希 `a2b0d1ac...` 的 lm-head-only 严格前沿。完整补丁、trace、质量结果和退出规则见 `candidates/reachable-backbone-gemv/summary.json`。

这轮对 agent 架构最重要的补充是：**源码变了不是可达性证据，局部快也不是生产证据，统计不显著更不是质量等价证据。** 候选必须依次通过运行时可达、同源整模转化、确定性、跨实现正确性和任务质量门；任何一层失败，都生成可重开条件明确的退出证书，而不是继续在同一参数族里消耗数十小时。

### 第十七轮：融合 lm-head 与 greedy argmax 可行，但整步上限只有 1.004x

严格前沿的 `lm_head` 已达到本机校准权重流屋顶的 99.31%，因此这次不再改 dot-product schedule，而是尝试改变算法边界：普通 vLLM greedy 路径会先物化 248,320 个 BF16 logits，再转成 FP32 后执行 argmax；候选保持已接受的逐行 FP32 累加和 BF16 rounding，在第一个 kernel 内只写每组局部最大值，随后用两级归约直接输出 token id。全零输入还专门验证了 tie 时必须选择最小 token id。

对 `BLOCK_N={4,8,16} × warps={4,8}` 六个点做交替测量后，所有候选在随机输入和全零 tie case 上都与生产路径 argmax 相同。最佳 CUDA Graph 点为 `BLOCK_N=16, warps=8`：完整 logits+argmax 为 2085.120 us，融合候选为 2052.301 us，局部 1.016x、只省 **32.819 us/token**。以严格前沿 8.079 ms TPOT 计算，整步预测仅 **1.0041x**；达到 1.03x 至少需要 235.3 us，本候选只覆盖 13.95%。

因此没有为了一个约 0.4% 的上限去侵入 model runner 和 sampler。该方向只有在能够省掉 lm-head 的大部分权重读取，而不仅是省掉约 1.5 MiB 的 logits cast/argmax 流量时才重开。机器可读退出证书为 `candidates/fused-lmhead-argmax/summary.json`，原始六点 CUDA Graph 测量为 `microbench_candidates/fused_lmhead_argmax_sm89.json`。

### 第十八轮：组合局部赢家没有线性叠加，热漂移不能算成优化收益

为了寻找更快的 BF16 质量前沿，把 segmented GDN projection 与较保守的 MLP-down 4-warp GEMV 同时启用，并用同一份已打补丁源码执行 control-candidate-candidate-control。两次候选分别为 7.8280 和 7.8798 ms TPOT，候选之间 6/6 token 一致；相对实现之间为 4/6 exact。

直接把四个进程平均会得到 E2E 1.0389x、TPOT 1.0428x，但第二个 control 的中文与翻译两例都异常升到约 8.75 ms，而其他 control case 仍在 8.05--8.11 ms。用未出现该异常的第一个 control 对两次候选，范围只有：

| 对照 | E2E 加速 | TPOT 加速 | 跨实现 exact |
|---|---:|---:|---:|
| C1 vs S1 | 1.0263x | 1.0321x | 4/6 |
| C1 vs S2 | **1.0220x** | **1.0253x** | 4/6 |

因此不能把末尾 control 的热/后台漂移计入收益，也不能把 segmented GDN 的约 4% 与 down GEMV 的约 3.6% 直接相加。保守同源证据未稳健跨过 1.03x 整模门槛，所以没有继续消耗约三分钟跑 GSM8K 512 题；候选记录为 `MEASURED_REJECT_UNSTABLE_MATERIALITY`，环境恢复到 lm-head-only 严格前沿。机器可读证据为 `candidates/combined-gdn-down/summary.json`。

这两个实验继续收紧了“最优”的含义：完整 logits 的物化确实不是理论必需，但它只占几十微秒；多个 memory-bound 局部赢家共享同一显存服务后也不能线性叠加。下一条可能产生物质性收益的算法必须减少 **权重读取本身**，例如带可验证上界的 vocabulary pruning / exact maximum-inner-product search，而不是继续减少小张量写回或把独立 kernel 的节省相加。

### 第十九轮：精确词表剪枝也无法避开权重读取

沿着上一轮的重开条件，本轮没有直接写 GPU kernel，而是先问一个更便宜、也更决定性的问题：对真实自回归 hidden state，带严格正确性证明的 maximum-inner-product 上界究竟能排除多少词表行？实验采集了六类自然提示各 32 个 token，共 192 个 Qwen3.5-0.8B BF16 decode state；对 248,320 行输出权重构建 256 个离线聚类，并使用分块 L2 residual bound 保证任何被跳过的 cluster 都不可能包含 top-1。

所有经验上界均有效，真实 winner cluster 也全部保留，但性能可行性完全不成立：

| 精确筛选口径 | 中位需读取权重行 | 加索引后的总字节比例 | 理想整步加速 |
|---|---:|---:|---:|
| 可执行 cluster bound | 99.9734% | 100.1828% | **0.99953x** |
| oracle 单行 norm bound | 100.0000% | 100.1953% | **0.99950x** |

第二行是更强的必要条件：它作弊使用已经算出的真实 top-1 分数作为 lower bound，再用每行 `||w_i||·||h||` 判断该行是否可能获胜。即便知道答案，这个上界仍不能删除任何一行，因此不是聚类数或 k-means 质量没有调好，而是这类范数界在本模型 hidden-state 几何上过松。实际 GPU kernel 还要付出 bound evaluation、索引、gather 和不规则访存成本，只会更差。

所以 exact cluster/norm pruning 已在写 kernel 之前用上界证书关闭，避免 agent 再花数小时扫 cluster、block 和 launch 参数。只有出现明显更紧的可证明索引，并在更大 hidden-state 集合上使 p90 总读取字节不超过 dense 的 75%，才允许重开。证据为 `models/sm89_exact_vocab_pruning_feasibility.json`，结案摘要为 `candidates/exact-vocab-pruning/summary.json`。

这进一步界定了 BF16 与量化路线：严格 BF16 若要继续取得数量级更大的收益，不能依赖通用 Cauchy-Schwarz 剪枝；而量化确实能直接减少权重流量，但公平基线必须同时切到相同量化格式的 vLLM，不能拿自研量化路径与 BF16 vLLM 比出一个虚假的倍数。

### 第二十轮：不量化权重，也能继续突破 BF16 访存下界

前一轮关闭词表剪枝后，真正剩下的矛盾是：`lm_head` 已接近“读取 508.6 MB BF16 权重”的带宽下界，继续改 GEMV 调度几乎没有空间，但减少读取字节通常会被叫作量化。这里找到了一条不同路线：**压缩 BF16 的表示，而不是降低 BF16 的精度**。

Qwen3.5-0.8B 的 `248320 × 1024` 输出权重共有 254,279,680 个 BF16 值，但只有 5,874 种不同位模式；指数只有 33 种，指数熵约 2.574 bit/value。候选把每个值的 sign+mantissa 原样存为 8 bit，再按 256 值一块存 4 bit 的“指数减块内最小指数”；指数跨度超过 15 的 1.486% 块原样回退 BF16。解码时在寄存器里重建完整 16 bit BF16 位模式，再转 FP32 做乘加。因此它不是 INT8/FP8/W4A16，也没有 scale、截断或近似：**每一个权重位都可逆还原**。

| 项目 | Dense BF16 | exact-packed BF16 |
|---|---:|---:|
| lm_head 存储 | 508,559,360 B | 393,940,992 B |
| 相对字节 | 100% | **77.462%** |
| 权重位精确 | 是 | **是** |
| 稳态 lm_head 中位延迟 | 2044.466 us | **1586.713 us** |
| 局部加速 | 1.000x | **1.2885x** |

这个结果需要一个容易被忽略的测量说明：packed kernel 同时使用更多整数/位运算，RTX 4060 Laptop 的核心频率从冷态爬升时，前几百次可从约 2.95 ms 慢慢降到约 1.99 ms。最终微基准使用 1,000 对交替 warmup 和 20 次交替采样，才得到上表的稳态 1.2885x。冷启动性能与稳态性能都是真实状态，不能只挑较快的一段；vLLM 的 CUDA Graph 编译和首次 cache 构建也不计入 steady-state decode。

完整模型通过环境开关接入同一份 vLLM 源码。先做 control-candidate-candidate-control，两个 control 平均 TPOT 7.1798 ms，两个 candidate 平均 6.8510 ms，提升 1.0480x；最保守的末尾 `C2 vs S2` 仍有 1.0390x。随后按正式门槛跑 6 个自然提示、每侧 3 次 warmup × 10 次测量、每例生成 128 token：

| 正式口径 | Control TPOT | Candidate TPOT | TPOT 加速 | E2E 加速 | token exact |
|---|---:|---:|---:|---:|---:|
| 当前已接受严格 BF16 前沿 → packed | 6.9951 ms | **6.6568 ms** | **1.0508x** | **1.0478x** | **6/6** |
| 当前 fresh stock → packed | 6.9205 ms | **6.6568 ms** | **1.0396x** | **1.0364x** | 4/6 |

第二行的 4/6 必须正确解释：fresh stock 与当前已接受前沿在 Python 与翻译用例上本来就走出了不同 token，说明 fresh AOT/cuBLAS 构建选择了另一棵 FP32 reduction tree。packed 权重本身仍逐 bit 精确，且在同源、同编译缓存的 accepted/packed 对照中达到 6/6。换句话说，“权重无损”不等于“任意两个浮点归约实现的中间 logits 位相同”。候选的最大 logits 差为 `6.1035e-05`，来自乘加顺序，不来自权重精度损失。

为了验证“只要压到 12 bit 就会快”这个假设，还实现了完全精确的 4,096 项 BF16 codebook + 12-bit index + fallback sibling。它同样约占 dense 的 77.61%，输出甚至可逐 bit 相同，但 table lookup 与 12-bit unpack 把 lm_head 拉慢到 4,388 us，只有 0.4665x。真正有效的不是压缩率单一指标，而是**压缩格式必须与 GPU 可并行解码的数据路径一致**；base+delta 能用规则位运算重建，codebook 的间接访问不能。

因此本轮把 exact-packed 晋级为这个受限范围的新严格 BF16 前沿：相对当前 stock 的直接实测约快 4%，相对此前严格前沿再快约 5%，不是“比 vLLM 快好多倍”，也不是全局理论最优证明。历史低功耗条件下的约 1.199x 与本轮 1.051x 没有在同一次受控实验中测量，禁止相乘后宣称累计 1.26x。当前原型还因为 embedding/lm_head 权重绑定而额外缓存约 375.691 MiB；生产化应在加载时直接构建 packed 表示，或者显式管理 tied weight，而不是永久保留两份输出权重。

这轮也给 agent 增加了一种更有“大局观”的搜索方式：当算子已经碰到 dense-byte roofline，不再围绕 block size 无界扫参，而是先统计真实权重信息熵，推导可逆格式的理论字节数；只对预测能跨过 3% 整步门槛的格式写 kernel；并并行构造一个压缩率相近但解码机制不同的负对照。机器可读汇总、补丁与全部证据位于 `candidates/exact-bf16-packed-lmhead/summary.json`。

### 第二十一轮：全骨干无损压缩的理论 1.205x，实测却只有 0.845x

`lm_head` 的 exact-packed BF16 成功后，一个自然假设是把同样的无损表示扩展到所有 decode projection。先扫描真实 checkpoint：活跃骨干 projection 的 dense BF16 共 995,229,696 B，exact-packed 后为 771,221,760 B，只占 77.492%。若错误地假设延迟与字节完全线性且解包免费，每 token 可省 1.133 ms，当前 6.657 ms TPOT 可乐观提升到 **1.205x**。这个上界足以通过 1.03x 物质性门，所以才进入 GPU 实测，而不是凭感觉提前否决。

但按真实 decode 顺序轮转所有层权重、确保每组工作集都超过 4060 的 32 MiB L2 后，结果反转：

| 路线 | dense 冷流 | 候选冷流 | 局部加速 | 投影整步 |
|---|---:|---:|---:|---:|
| scalar exact unpack，五类主要 projection 最佳点求和 | 4936.909 us | 5845.370 us | **0.8446x** | **0.8799x** |
| metadata hoist + padded-M=16 Tensor Core，代表性 attention QKV | 312.710 us | 690.790 us | **0.4527x** | **0.9463x**（仅该组） |

两条路线都逐位验证了权重重建，失败原因不是精度或实现没跑到，而是硬件数据路径不匹配。`lm_head` 是单个 508.6 MB 的巨大 DRAM stream，规则位运算能被少读 22.5% 权重流量掩盖；骨干矩阵更小、形状更多且逐层重复，cuBLAS 的 shape-specific reduction 已很高效，整数解包、metadata、M=1 填充到 M=16 和同步成本反而成为新瓶颈。也就是说，“同一格式在一个大矩阵上赢”不能外推为“在全模型每个 GEMV 上都赢”。

为了避免把微基准负结果误判成候选不可达，又对当前 exact-packed 前沿做了 Nsight Systems 2025.5.1 完整图采样：期望 192 次 packed lm-head，实际观察到 192 次，可达性通过。profile 下 `cublas_gemvx` 占 GPU kernel time 的 52.74%，packed lm-head 占 30.90%，说明骨干确实是下一大热点；但 profiler 把 packed kernel 扰动到约 2.812 ms，因此这些比例只用于机会排序，不能与非 profile 的 1.587 ms 稳态值混算。

这一轮还直接检验了“把两个局部赢家叠加”：保持 exact-packed lm-head 开启，只切换此前通过有限 GSM8K 质量筛选的 segmented GDN。在同源、6 提示、每例 3 warmup × 10 次、固定 512 MiB KV cache 下，组合从 6.6548 ms 退化到 6.7903 ms TPOT，只有 **0.9801x**，并且只有 2/6 token exact；六个提示全部变慢。此前独立测得的约 4% GDN 收益不能与 packed lm-head 的约 5% 相乘，因为它们共享显存带宽、调度、功耗状态和浮点归约路径。

因此严格 BF16 前沿仍是“只对 lm-head 做 exact packing”。按本机 248.8785 GB/s 校准带宽，当前活跃权重流量约 1,426,884,288 B，单纯服务这些字节的 floor 约 5.733 ms；即使不可能地删除所有非权重时间，相对 6.657 ms TPOT 的绝对上限也只有约 **1.161x**。这不是全局理论最优证明，但已给当前“保持 BF16 权重位、batch=1、同模型同解码语义”的局部搜索空间画出很窄的边界。下一次重开全骨干压缩，必须出现原生 compressed-BF16 dot、在冷轮转权重上逐 shape 超过 cuBLAS 的解包路径，或能同时删除额外 producer-consumer 流量且预测整步至少 3% 的新架构。

工程上还暴露了一个需要在生产化前修复的问题：当前原型在构建 packed cache 时短暂同时保留 tied dense 权重与中间张量，vLLM 自动 KV cache profiling 会把这个峰值当作常驻占用，甚至算出负的可用缓存。实验通过固定 512 MiB KV cache 隔离了该问题；正式实现应在 CPU/load-time 分块打包并转移 ownership，不能依赖这个运行时绕过。

机器可读退出证书分别为 `candidates/exact-bf16-packed-backbone/summary.json` 和 `candidates/exact-lmhead-segmented-gdn/summary.json`。它们把“理论上值得试—实测为什么失败—什么条件才允许重开”写死，正是 agent 避免卡在局部死路的机制。

### 第二十二轮：vLLM 不是理论最优；公平的量化比较要优化“量化后的剩余瓶颈”

用户追问“既然只能靠量化，vLLM 自己量化不是更快吗”。这个问题的前半句需要先纠正：严格 BF16 路线并非完全没有空间，当前 exact-packed lm-head 就是在不丢失任何 BF16 权重位的前提下取得了约 4%--5% 的整模提升；但在已经接近权重访存下界后，继续期待 batch=1 解码翻倍并不符合当前测得的物理预算。量化可以直接减少 backbone 权重流量，不过公平基线必须是 **同一种量化模式的 vLLM**，而不是拿自研量化去打 BF16 vLLM。

本轮先筛了 vLLM 自带在线量化路径。相邻短测中，FP8 per-tensor、per-channel 分别只提升约 1.6% 和 1.0% TPOT，没有跨过 3% 物质性门；`fp8_per_block` 选择 Marlin 的 weight-only kernel，约提升 10.8%，因此晋级。源码运行日志同时显示它转换了 164 个 backbone linear layer，但 tied embedding/lm-head 仍保留 BF16。这提供了一个明确的残余机会：保留 vLLM 的调度、CUDA Graph、attention/recurrent 和 Marlin backbone，只把未量化的大词表输出层换成上一轮已经验证过的 exact-packed BF16 kernel。

正式实验固定 `max_num_seqs=1`、512 MiB KV cache、六类自然提示、128 输出 token，并按 C-S-S-C 顺序对“BF16 backbone + exact head”和“FP8 per-block backbone + exact head”各跑两次、每例 3 warmup × 10 trials：

| 口径 | BF16+exact head | FP8-block+exact head | 加速 |
|---|---:|---:|---:|
| mean E2E | 1112.788 ms | **995.911 ms** | **1.1174x** |
| mean TPOT | 8.3936 ms | **7.5539 ms** | **1.1112x** |
| 四个 control/candidate 交叉 TPOT 范围 | — | — | **1.1028x--1.1196x** |
| 重复运行自身 token exact | 6/6 | 6/6 | 确定性通过 |
| BF16 与 FP8 跨模式 token exact | — | 0/6 | 非严格路线 |

所有六类提示在每个交叉比较中都变快，而且候选的中位 graphics clock（915/930 MHz）还低于 control（945/960 MHz），因此不能把收益解释为候选恰好跑在更高频率。不过笔记本绝对延迟仍明显受功耗状态影响，正式结论使用 10.3%--12.0% 的交叉区间，不采用早期短测里最高的 27%。

量化路径不能只看速度。冻结 512 道 GSM8K 后，BF16 control 为 45/512（8.79%），混合候选为 40/512（7.81%），点估计下降 0.98 个百分点；配对结果为 control-only 15、candidate-only 10，McNemar 双侧精确检验 `p=0.424`。这没有检测到统计显著差异，但答案一致率只有 65.23%，因此它只被标记为 **limited latency-first discovery frontier**，不升级为质量等价或生产默认。严格默认仍是 exact-packed BF16。

这一轮得到的架构原则比某一个倍率更重要：

1. vLLM 是强工程基线，不是对任意模型、GPU、batch 和质量目标都成立的理论最优。
2. “参数量相同”不等于计算成本相同；格式字节数、kernel 是否匹配硬件、未量化层、调度与上下文分布都会改变瓶颈。
3. 自研量化若不与相同量化的 vLLM 对照，倍率没有意义；真正可持续的优势是复用 vLLM 已做好的部分，再根据 profile 找出它在特定模型上留下的热点。
4. 优化目标应是 Pareto 前沿，而不是一个无条件的“最快”：严格 BF16 档保语义，延迟优先档用质量预算换约 11% TPOT，二者都必须明确证据范围。

完整证据和复现入口见 `candidates/fp8-block-exact-lmhead/summary.json`；其他在线量化 sibling 的有界退出记录见 `candidates/vllm-online-quantization-sm89/summary.json`。

### 第二十三轮：同量化 vLLM 上再快 9.35%，但不能外推到多并发

上一轮已经证明混合 FP8 路线快于 BF16，但“增量到底来自 vLLM 自带量化还是自研 lm-head”仍缺正式隔离。本轮把两边都固定为相同的 `fp8_per_block` Marlin backbone、相同源码哈希和相同 512 MiB KV cache，只切换 stock BF16 lm-head 与 exact-packed BF16 lm-head，再执行 C-S-S-C、六提示、每例 3 warmup × 10 trials：

| 同量化正式对照 | 原生 vLLM FP8 | FP8 + exact head | 加速 |
|---|---:|---:|---:|
| mean E2E | 686.649 ms | **629.648 ms** | **1.0905x** |
| mean TPOT | 5.1190 ms | **4.6812 ms** | **1.0935x** |
| 四个交叉 TPOT 范围 | — | — | **1.0933x--1.0937x** |
| 六提示跨实现 token exact | — | **每一对 6/6** | 通过 |

候选的 graphics clock 为 2610/2610 MHz，对照为 2625/2610 MHz，因此 9.35% TPOT 增量不是更高核心频率造成的。更强的 512 题 GSM8K 隔离中，两边都是 40/512，答案一致 512/512，生成 token 也一致 512/512。至此可以把贡献拆开：backbone FP8 决定相对 BF16 的质量变化，而 exact-packed lm-head 在同一 FP8 数值合同上提供额外、可重复的 batch-1 加速。

随后没有直接把 M=1 结果外推到服务并发，而是实现了一个让同一 packed 权重 tile 在 CTA 内服务 2--4 行输入的小批量 sibling。孤立 CUDA Graph 微基准非常漂亮：batch 1/2/4 分别约 1.302x、1.295x、1.293x，batch 8 才降到 0.933x；所有最佳点 argmax 一致。但接入 `max_num_seqs=8` 的 vLLM 服务曲线后结果完全反转：

| 并发 batch | stock wall | small-batch packed wall | stock/candidate |
|---:|---:|---:|---:|
| 1 | 362.183 ms | 749.664 ms | **0.483x** |
| 2 | 375.764 ms | 763.663 ms | **0.492x** |
| 3 | 385.669 ms | 796.334 ms | **0.484x** |
| 4 | 410.332 ms | 806.530 ms | **0.509x** |
| 8 | 468.441 ms | 465.631 ms | 1.006x（回退 stock） |

为了判断是不是新增 M=2--4 kernel 单独造成的，又撤回该扩展，只保留此前成功的 M=1 exact kernel。`max_num_seqs=8` 下 batch=1 仍从 stock 362.183 ms 退化到 873.838 ms，而 batch=2--8 因选择器回退 stock 基本回到原曲线。batch=8 是同进程负对照，说明不能把整次候选进程简单归因为机器整体变慢。当前证据能证明“多并发配置不组合”，但尚不能在没有 graph timeline/clock trace 的情况下武断区分 CUDA Graph 边界、运行时 shape/layout 与功耗状态各占多少。

因此前沿现在是**条件化选择**，而不是一个开关覆盖所有场景：

- `max_num_seqs=1`、batch-1 latency：FP8 per-block + exact-packed head，相对同量化原生 vLLM 约 1.094x，当前赢家。
- `max_num_seqs>1` 或持续并发吞吐：关闭当前 exact head，保留原生 vLLM FP8；small-batch sibling 已实测拒绝。
- 若要重开小批量路线，必须先证明 kernel 位于预期 CUDA Graph、归档真实运行 shape，并在锁频或在线自适应选择下让完整服务曲线逐点不劣于 stock；不能再用孤立 GEMM 的 1.29x 作为晋级理由。

这也是 agent 架构需要学习的更高层规则：优化结果必须是 `(硬件, 模型, 数值合同, batch/concurrency, runtime config)` 的策略表；所谓“最优 kernel”可能只是某一个服务点的最优，调度器应按实测前沿选择，而不是把单点赢家全局启用。

机器可读正式隔离为 `comparisons/vllm_fp8_native_vs_exact_qualification.json`，多并发退出证书为 `candidates/exact-bf16-packed-lmhead-small-batch/summary.json`。

### 第二十四轮：画像驱动的 INT8 lm-head，把同量化 vLLM 提升到 1.219x

对上一轮 `FP8 per-block backbone + exact-packed BF16 lm-head` 前沿做 Nsight Systems 2025.5.1 画像后，机会排序变得非常集中：exact-packed lm-head 每生成 token 约占 1.528 ms，在观测 CUDA kernel 时间中占 **90.17%**；全部 Marlin backbone 和其他 kernel 合计只占余下约 9.83%。因此没有继续在 attention、采样或几十微秒的小 kernel 上试错，而是直接建立有损 lm-head 压缩候选族。

候选用 signed INT8 保存 248320×1024 输出投影，沿 K 维每 32/64/128/256/1024 个值共享一个 FP16 scale，并在 batch-1 Triton GEMV 内完成反量化与 FP32 reduction。五种 group、每种四个可编译 schedule 的有界微基准显示：group-32/128/1024 相对 exact-packed head 分别最快约 1.459x/1.525x/1.540x。没有直接选择最快的 per-row（1024）量化，而是选择误差更低的 group-32 进入端到端，这是把质量预算显式放入搜索目标，而不是只追逐局部速度。

正式 C-S-S-C 固定相同源码、相同 FP8 per-block Marlin backbone、`max_num_seqs=1`、512 MiB KV cache、六类自然提示、128 输出 token，每个状态每例 3 warmup × 10 trials：

| 正式对照 | FP8 + exact BF16 head | FP8 + INT8 group-32 head | 加速 |
|---|---:|---:|---:|
| mean E2E | 629.299 ms | **570.171 ms** | **1.1037x** |
| mean TPOT | 4.6798 ms | **4.2010 ms** | **1.1140x** |
| 四个交叉 TPOT 范围 | — | — | **1.1104x--1.1176x** |
| 相对原生 vLLM FP8 TPOT | 5.1190 ms | **4.2010 ms** | **1.2185x** |

四个进程的 graphics clock 中位数均为 2610 MHz。自然提示有 5/6 完整 token 相同，说明它不是严格等价替代。冻结的 512 题 GSM8K 上，exact-head control 为 40/512，INT8 group-32 为 42/512，答案一致 495/512、token exact 493/512，McNemar 双侧精确检验 `p=0.625`。这只能说明当前任务筛选未检测到退化，不能证明广泛质量等价。

所以现在有三档明确策略，而不是一句“量化更快”：

- 严格同输出、单序列：FP8 backbone + exact-packed BF16 head，相对原生 vLLM FP8 为 1.094x。
- 经目标任务质量校准、单序列延迟优先：FP8 backbone + INT8 group-32 head，相对原生 vLLM FP8 为 **1.219x**，但必须承认有损。
- 多序列/吞吐优先：保留原生 vLLM FP8；已有 custom small-batch 路径没有在完整服务中组合成功。

这轮也验证了 agent 应有的闭环：先用 profile 把 90% 热点找出来，再以格式/架构候选族覆盖速度—质量 Pareto，局部筛选后才做端到端 C-S-S-C 与 512 题质量门。完整证据见 `candidates/int8-groupwise-lmhead/summary.json` 和 `comparisons/vllm_fp8_exact_vs_int8head_g32_qualification.json`。

### 第二十五轮：量化只做候选召回，BF16 复核把 1.199x 与同输出重新结合

纯 INT8 lm-head 虽快，但上一轮只有 493/512 条 GSM8K token 流与 exact head 一致。本轮没有在“是否量化”之间二选一，而是把输出头改成两阶段决策：先用 INT8 扫描全部 248320 个词表行，只取近似 top-k；随后从原始 BF16 tied embedding 中读取这些候选行，以 FP32 reduction 精确重算，并让 greedy argmax 只在精确复核值中选择。

先测试 group-32。top-2 在六类自然提示中有一例于第 92 个 token 分叉，因此拒绝；top-4 达到 6/6×128 token 和 GSM8K 512/512 token exact。Nsight 又显示 top-4 的 BF16 复核只有 0.00124 ms/token，远小于 INT8 全词表扫描，于是继续扩大候选而降低量化元数据：最终选择每行一个 scale（group-1024）配 top-128。它将 INT8 缓存压到 dense BF16 的 50.10%，而 top-128 BF16 复核仍只有 0.00229 ms/token。

同源码 C-S-S-C 正式结果如下：

| 正式对照 | FP8 + exact BF16 head | FP8 + INT8 per-row scan + BF16 top-128 | 加速 |
|---|---:|---:|---:|
| mean E2E | 632.805 ms | **582.469 ms** | **1.0864x** |
| mean TPOT | 4.6760 ms | **4.2695 ms** | **1.0952x** |
| 四个交叉 TPOT 范围 | — | — | **1.0937x--1.0967x** |
| 相对原生 vLLM FP8 TPOT | 5.1190 ms | **4.2695 ms** | **1.1990x** |

四个交叉对照均为 6/6×128 token exact；冻结 GSM8K 512 题上，两边都是 40/512，答案与整段 token 流均为 512/512 一致。Nsight 对 192 个生成步骤观测到 192 次 INT8 scan 和 192 次 exact rerank；scan 为 0.99290 ms/token、占 83.24% GPU kernel 时间，top-128 rerank 仅 0.00229 ms/token、占 0.19%。这证明实际运行的正是两阶段路径，也说明下一步若要继续加速，目标仍应是 full-vocab scan，而不是纠结复核 kernel。

但这里必须区分三种“相同”：它对本轮自然提示和 GSM8K 是**逐 token 实证相同**；它不是全输入的数学保证；它也不是完整 logits 相同，因为 shortlist 外被置为 `-inf`。因此它只适用于经验证的 batch-1 greedy argmax，不可直接用于 logprobs、top-p 或随机采样。要升级为严格保证，需要给量化误差建立保守上界：只有当近似 shortlist 的边界间隔足以证明 BF16 winner 必在其中时走快路，否则自动回退完整 exact head。

当前条件化前沿因此变成四档：严格全输入合同用 exact-packed BF16；有界实证同输出 greedy 用 per-row INT8 + BF16 top-128，约为原生 vLLM FP8 的 **1.199x**；允许目标任务校准有损时用 group-32 INT8，约 **1.219x**；多序列服务继续用原生 vLLM FP8。完整证据见 `candidates/int8-bf16-shortlist-rerank/summary.json`。

### 第二十六轮：vLLM 量化并非自动更快，W4A16 找到更强内核但尚未晋级

这一轮直接检验“既然 vLLM 也能量化，量化 vLLM 是否必然更快”。答案是否定的：在 RTX 4060 Laptop 上，vLLM 明确报告该设备没有它所需的原生 FP8 计算路径，`fp8_per_block` backbone 实际选择 Marlin weight-only；与此同时 tied `lm_head` 仍保留 BF16。这也是此前还能优化输出头的原因——对照并不是未量化 vLLM，而是 **Marlin FP8 weight-only backbone + BF16 head**。

首先测普通 Triton packed INT4。最快点只需 dense BF16 约 25.78% 的存储字节，但扫描仍为 1.838 ms，慢于已有 INT8 的 0.993 ms；低 4 bit 带来的字节节省被 nibble 解包、符号扩展和反量化指令吃掉，因此停止继续调 block/warp。随后复用 vLLM Marlin 的原生 W4A16 路径，完整 248320×1024 扫描达到 **0.894 ms**，相对 exact-packed BF16 的 2.956 ms 为 **3.307x**；8/8 随机向量的 BF16 真正赢家都在近似 top-128 中，最坏近似排名仅为 3。W4A8-INT8 反而为 2.574 ms，说明“bit 更低”不能替代真实 shape 上的测量。

把 W4A16 scan 与 BF16 top-128 精确复核接入完整 vLLM 后，六类自然提示均与原生路径保持 128/128 token 相同。在当前笔记本低功耗状态下，原生与候选分别为 8.626 和 5.994 ms/token，候选约 **1.439x**；原生和候选中位核心频率分别为 945 和 862.5 MHz，所以收益不是候选频率更高造成的。但这组结果不能与此前 2610--2625 MHz 下的 5.119 ms 原生前沿交叉比较：当前 GPU 被限制在约 26--31 W、780--975 MHz。候选因此只记为强内核和低功耗端到端通过，尚不替代已经完成高频 C-S-S-C 与 GSM8K-512 的 INT8+BF16 前沿。

当时的全局画像曾写成：INT8+BF16 前沿 TPOT 为 4.260 ms，而 Nsight `KERNEL` 表总时长除以生成步数只有 1.193 ms，因此约 **3.067 ms/token（72.0%）** 位于 kernel 外。第三十六轮复核发现这个推断错误：当前 Nsight trace mode 不把 CUDA Graph replay 的子 kernel 展开到 `KERNEL` 表，而是另以 186 条 `GRAPH_TRACE` 记录回放主体。旧的 72% 结论已撤回；修正后的运行时边界见第三十六轮。

完整记录见 `candidates/marlin-w4-bf16-shortlist-rerank/summary.json`、`candidates/ordinary-triton-int4-recall/summary.json`、`models/vllm_fp8_rerank_decode_cadence_bound.json` 和 `comparisons/vllm_fp8_native_vs_marlin_w4_rerank_low_power_screen.json`。

### 第二十七轮：W4 让双 token 输出头接近“免费”，但 MTP-1 仍应拒绝

为了检验能否利用设备驻留的多 token 验证减少逐 token 调度，本轮先扩展 W4 shortlist/rerank 原型，使它同时支持 M=1 和 M=2。M=2 完整词表扫描的 Marlin W4A16 中位耗时为 **0.803 ms**，而同 shape 的 `torch` BF16 linear 为 2.842 ms，约快 **3.539x**；8/8 个随机向量的 BF16 真正赢家仍全部进入 top-128，实测两行赢家近似排名均为 1。相较 M=1 的 0.894 ms，M=2 没有翻倍，证明 Marlin 确实复用了输出权重流量。

但是局部收益没有转化为完整推理收益。相同 `fp8_per_block` 主干和 W4+BF16 精确回排下，六类自然提示、64 输出 token 的低功耗筛选结果为：

| 路径 | 加权 TPOT | 加权 E2E | 相对结果 |
|---|---:|---:|---:|
| 普通单 token decode | **6.216 ms** | **431.866 ms** | **1.114x faster** |
| 原生 MTP-1 | 6.922 ms | 488.484 ms | reject |

MTP 总接受率为 161/219（73.52%），且只有 3/6 个样例与普通路径保持整段 token 相同。虽然 exact BF16 rerank 消除了输出头候选内部的量化误差，但 M=2 target backbone 的矩阵 shape 和归约树也会改变上游 hidden state；因此不能由“输出头精确复算”推出 M=1/M=2 的全链路 token 等价。性能上，MTP proposer、双 token target 验证、拒绝处理及额外 runtime 工作仍超过少一次调度带来的收益。

这轮体现的不是“保守地少做一种优化”，而是先用 M=2 微基准验证必要条件，再用一次短端到端实验否定充分性。由于性能门和 token-identity 门同时失败，不再浪费数小时做 3×10 或质量集复验；Marlin W4 继续作为普通 batch-1 greedy 候选，MTP-1 在当前 `(Qwen3.5-0.8B, RTX 4060 SM89, vLLM 0.28.1, concurrency=1)` 点关闭。证据见 `comparisons/vllm_fp8_marlin_w4_rerank_mtp1_screen.json`。

### 第二十八轮：公平 GPTQ-Marlin 对照证明“vLLM 量化”仍不等于局部最优

为了直接回答“如果 vLLM 也量化，它不是应该更快吗”，本轮下载并校验了同一 Qwen3.5-0.8B 的历史 GPTQ W4A16、group-size 128 checkpoint。971,213,992 字节的 `model.safetensors` 与远端 LFS SHA256 一致，共有 923 个张量，其中 150 个 `qweight`；加载后 vLLM 报告模型权重约 0.85 GiB，并明确选择 `MarlinLinearKernel`。这建立了真实 W4 对照，不再把 BF16/FP8 与 W4 淵称为同一路径。

checkpoint 最初不能由 vLLM 加载，但这是可修复的技术问题：文件中 full-attention 的 q/k/v/o projection 已经量化，`quantization_config.modules_in_block_to_quantize` 却只列出 linear-attention 和 MLP。vLLM 因而按 BF16 创建融合 QKV 层，再收到 GPTQ `g_idx` 而失败。补上 `self_attn.q_proj/k_proj/v_proj/o_proj` 四个元数据条目后，无需改任何权重即可成功加载。修补记录在 `candidates/gptq-marlin-backbone-fair-baseline/quantization_config_self_attn_fix.patch`。

在六类自然提示、每条 128 greedy token、2 次 warmup + 5 次测量的低功耗本地对照中，库存 GPTQ-Marlin vLLM 为 **7.688 ms/token（130.1 tok/s）**。相对此前同属低功耗筛选的 FP8 stock 8.626 ms/token，它方向性快约 **1.122x**，但两者没有做进程交错，不能冒充正式因果资格赛；更不能拿它与 2610 MHz 高频 FP8 的约 5.119 ms/token 直接判量化优劣。量化收益不大的一个明确原因是该 checkpoint 的 tied `lm_head` 未量化，24.8 万词表输出投影仍是 BF16，而 Marlin 的反量化与小矩阵调度也有固定成本。

在相同 GPTQ-Marlin 骨干上启用 W4 全词表召回 + BF16 shortlist 精确复核，top-128 候选达到 **5.049 ms/token（198.0 tok/s）**，相对库存 GPTQ vLLM 快 **1.523x**；候选中位核心频率和功耗还更低（825 MHz、23.835 W，对照为 915 MHz、28.79 W），所以方向性的收益可信。但它只在 2/6 条完整序列上与库存 GPTQ 逐 token 相同，因此性能门通过、质量门失败，不能晋级生产冠军。

将 shortlist 从 128 扩到 512 后，六条序列的分叉位置完全不变，单轮 TPOT 仍约 5.007 ms。随后在 eager 模式逐 token 计算完整 BF16 参考 logits，BF16 赢家的 W4 最差近似排名只有 zero-based 1，证明扩大 shortlist 不是正确修复方向。当前剩余问题被收窄到 torch.compile/CUDA Graph 下的执行或浮点数值语义；下一轮应做 compiled-no-graph 与 graph 的正交隔离，或在低 margin 时回退完整 BF16，而不是继续扫 top-k。

因此更准确的结论是：**vLLM 是很强的通用执行框架，量化也是必要候选，但“vLLM + 量化”仍不是这张卡、这个模型、这个 batch 的自动最优解。** 我们已经实测到其上仍有约 1.52x 的局部空间，同时也实测到追求速度会碰到真实的质量边界。机器可读证据见 `comparisons/vllm_gptq_marlin_w4_stock_vs_specialized_head_low_power.json`。

### 第二十九轮：在同量化 vLLM 上复现 1.55x，并把“速度冠军”和“无损冠军”分开

上一轮的 Triton shortlist 归约在 CUDA Graph 下只保持 2/6 条自然提示逐 token 一致。本轮先正交关闭 CUDA Graph：torch.compile 保留时恢复 6/6、768/768 token，但 TPOT 退化到约 19.68 ms，说明关闭 Graph 不是可用修复。单独捕获候选算子又能与 eager 完全一致，进一步把问题收窄为真实 hidden state 上不同矩阵归约路径的数值次序，而不是 Graph 回放破坏了缓存。

修复将 top-128 的 BF16 重排从自写 Triton reduction 改为 batch-1 `torch.mv`（M=2 安全回退 `torch.bmm`），以更接近库存输出头的 GEMV 归约语义。两个完全独立的新进程和新编译缓存都得到 6/6、768/768 token 一致，TPOT 分别为 5.027 ms 和 5.025 ms。随后 2 次 warmup + 5 次测量的资格轮得到 **4.959 ms/token（201.7 tok/s）**；对照 stock GPTQ-Marlin 的 **7.688 ms/token（130.1 tok/s）**，同量化、同引擎下提升 **1.550x**。候选测量中位核心频率还更低（825 对 915 MHz），因此不是升频造成的假收益。

但 512 道 GSM8K、每题最多 64 token 的配对测试揭示了必须保留的边界：两边准确率都为 42/512，McNemar 双侧精确检验 p=1.0，然而答案只在 466/512 一致，完整 token 序列只在 457/512 一致。这说明 top-128 召回加 BF16 重排在冻结的吞吐提示上可复现逐 token 一致，在更广输入上则仍是近似语义。它可以晋级为**显式 opt-in 的 batch-1 greedy 极速模式**，不能替换库存 vLLM 默认路径，也不能宣称数学等价或理论最优。

这轮也修正了比较口径：agent 不应试图从零重写 vLLM 后拿自己的量化版本去比较 BF16 vLLM；公平目标是先采用 vLLM 的 GPTQ-Marlin，再根据新瓶颈分布只替换残余热点。本例中主干 W4 后，未量化的 `248320 x 1024` tied `lm_head` 成为突出瓶颈，局部结构特化才产生额外 1.55x。机器可读证据见 `comparisons/vllm_gptq_marlin_w4_stock_vs_mv_rerank_qualification.json` 与 `comparisons/vllm_gptq_marlin_w4_stock_vs_mv_rerank_quality_gsm8k_n512.json`。

### 第三十轮：W4-only 更快但不是更优，质量门阻止错误晋级

在继续放宽数值契约前，先尝试了严格 BF16 输出头的双基准指数编码：用一位选择两个 block-local base、三位保存 delta，仍维持每值四位指数编码。对真实 2.54 亿个权重值扫描后，最佳 256-value block 的总字节占比为 77.512%，略差于现有单基准方案的 77.462%；尚未计算额外解码指令就已经输掉流量门，因此按照成本门直接停止，没有编写 GPU 内核。静态证据见 `candidates/exact-bf16-packed-lmhead/dual_base_static_screen.json`。

为了测清“既然 vLLM 量化更快，为什么不把输出头也直接 W4”，本轮在相同 GPTQ-Marlin 主干上移除 BF16 top-128 重排，只保留 W4 全词表扫描。一次新缓存筛选得到 **4.807 ms/token（208.0 tok/s）**，相对 stock GPTQ vLLM 为 **1.599x**，但相对上一轮 BF16 重排候选只多 **3.17%**。六类自然提示全部分叉，只有 57/768 token 与 stock 相同。

由于额外收益刚过 3% 材料性门槛，又运行了相同冻结索引的 512 道 GSM8K。stock、BF16 重排和 W4-only 都恰好答对 42/512，但 W4-only 相对 stock 只有 200/512 答案一致、167/512 序列一致；配对结果包含 22 道 stock-only 正确和 22 道 candidate-only 正确。总分相同只是相互抵消，不能解释为能力保持。相对 BF16 重排也只有 192/512 答案一致。

因此 W4-only 在正式重复性能资格赛前被淘汰：它为了额外约 3.2% 吞吐大幅替换模型行为，不位于实用的风险收益前沿。保留实现开关和原始证据只用于复现边界，默认关闭。这个结果同时说明 agent 的目标函数不能只有 tok/s；必须把配对行为稳定性纳入晋级门，否则它会把“换了一个碰巧同分的模型”误判成算子优化成功。证据见 `comparisons/vllm_gptq_marlin_w4_scan_only_boundary.json`。

### 第三十一轮：vLLM 量化已经更快，但最优路径随并发变化

本轮不再用单请求结果回答“有没有超过 vLLM”，而是在同一个 vLLM 0.28.1、同一 Qwen3.5-0.8B、同一张 RTX 4060 Laptop 上测量 batch 1/2/4/8 的服务曲线。每档生成 64 个 greedy token，1 次 warmup、3 次测量，测量轮交替正序和逆序；运行 stock→candidate→stock 夹心对照。事后审计发现当时只设置了 `TORCHINDUCTOR_CACHE_DIR`，而这版 vLLM 的 AOT 实际仍位于默认 `~/.cache/vllm`；候选日志确实报告直接载入该 AOT，末次 stock 则重新编译。因此启动时间不进入吞吐比较，并对 steady-state 性能采用两次 stock 中更快者作为保守基线，但这组记录不能再表述为 fresh-cache 隔离证明。基准工具随后新增 `VLLM_CACHE_ROOT` 记录和强制 guard，避免复发。

| batch | vLLM BF16 | vLLM GPTQ-Marlin 两次 | GPTQ + W4/BF16 rerank | rerank / 较快 GPTQ | rerank / BF16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 97.9 tok/s | 119.8 / 124.7 tok/s | **176.9 tok/s** | **1.419x** | **1.807x** |
| 2 | 185.0 tok/s | 274.3 / 281.8 tok/s | **316.5 tok/s** | **1.123x** | **1.711x** |
| 4 | 356.0 tok/s | **484.8 / 546.7 tok/s** | 472.8 tok/s | 0.865x | 1.328x |
| 8 | 670.0 tok/s | **883.2 / 917.4 tok/s** | 888.7 tok/s | 0.969x | 1.326x |

这直接证明两件事。第一，用户的直觉正确：库存 vLLM 的 GPTQ-Marlin 在所有实测并发上都比库存 BF16 快；同参数量不等于同流量，W4 每个权重需要搬运的字节显著少于 BF16，代价是解包、反量化和不同数值语义。第二，“vLLM 已经量化”仍不代表每个 shape 都达到局部最优。batch 1 时，大词表输出头的单行权重流量无法摊薄，W4 全扫描加 BF16 top-128 回排相对两次 stock 中更快者仍快 1.419x；batch 2 只剩 1.123x，而 batch 4/8 的库存矩阵路径已经把权重读取摊到多行，特化收益消失。末次 stock 在 batch 4 比首次快约 12.8%，也证明单次非交错数字不适合用来宣称 SOTA；这里用保守包络决定策略。

因此生产策略不应选择一个永久冠军：batch 1 可显式选择 approximate rerank，batch 2 仅在接受其质量契约且 15% 收益有价值时选择，batch 4 以上使用库存 GPTQ-Marlin。严格 BF16 是另一条数值契约，必须单列，不能因为参数数目相同就与 W4 混称公平速度冠军。

同缓存对照中 batch 1/2 分别达到 192/192、384/384 token 一致；batch 4/8 即使候选 shape guard 已回退库存路径，跨进程仍只有 721/768、1433/1536 token 一致。这说明后两档差异来自批量 GPU 归约和调度的数值非确定性，不能归因给未被调用的候选。另一方面，上一轮 GSM8K-512 的 457/512 序列一致仍是更强质量证据，所以本轮小集合完全一致也不能把 approximate 模式升级为无损默认。

对 agent 架构的含义是：搜索状态必须包含 workload shape 和精度契约，先测服务曲线，再生成分段策略；“某个算子在 M=1 最快”不再允许推出“模型推理全局最快”。机器可读证据见 `comparisons/vllm_bf16_gptq_rerank_service_frontier.json`，原始记录见对应的 `traces/vllm_*service_curve*.json`。

### 第三十二轮：严格 BF16 候选必须进入权重加载和预热生命周期

量化前沿明确后，本轮回到严格 BF16：候选不改变任何 BF16 权重 bit，而是把输出头按 256 值分块，用 8-bit sign+mantissa、4-bit block-local exponent delta 和约 1.49% dense fallback 可逆存储。旧原型在 batch-1 专用验证中有约 4--5% 端到端收益，但它在第一次调用时才生成 375.691 MiB packed cache，不满足服务部署要求。

把旧原型直接放入 `max_num_seqs=8` 的通用引擎后，active batch 1 只有 62.0 tok/s，远慢于同源 stock 的 101.5 tok/s；日志同时显示 packed cache 和 Triton JIT 都发生在 engine ready 之后。第一次修复误接到普通 Linear 的 `process_weights_after_loading`，但 Qwen3.5 的输出头与 embedding tied，不经过该路径，因此保持不可达并被撤销。真正的接入点是 `UnquantizedEmbeddingMethod.process_weights_after_loading`：它拥有 tied 权重，在这里完成 pack，并用一行零输入提前 JIT 内核。

最终候选与末次 stock 使用彼此独立且强制校验的 `VLLM_CACHE_ROOT`；首次 stock 发生在 guard 加入前，因此只把它作为更快、更不利于候选的保守 stock 包络。batch 1/2/4/8、每档 1 warmup + 3 trials 的通用服务曲线得到：

| active batch | 两次 stock BF16 | load-pack-warm 候选 | 相对较快 stock | 策略 |
|---:|---:|---:|---:|---|
| 1 | 101.5 / 97.1 tok/s | **110.9 tok/s** | **1.093x** | weight-exact 可选候选 |
| 2 | **191.8 / 187.3 tok/s** | 186.2 tok/s | 0.971x | stock |
| 4 | **413.1 / 342.0 tok/s** | 356.3 tok/s | 0.863x | stock |
| 8 | **696.7 / 658.8 tok/s** | 671.2 tok/s | 0.963x | stock |

生命周期消融同样关键：lazy first-use 为 62.0 tok/s，只在加载时 pack 后为 110.6 tok/s，再增加加载期 JIT 为 110.9 tok/s。也就是说，pack 前移修复了约 1.79x 的集成退化；JIT 前移主要消除首请求尖峰，steady-state 只变化约 0.3%。候选报告模型显存从约 1.53 GiB 增到 1.91 GiB，因为原型同时保留 tied dense 权重和 packed cache；生产版应把 packed 表示放入 checkpoint 或明确管理双表示。

数值契约也被进一步拆清：`max_num_seqs=1` 诊断中候选与 stock 为 320/320 token 一致且快 1.123x；通用 `max_num_seqs=8` 的 active batch 1 则为 187/192 token 一致。权重重建逐 bit 精确，但 FP32 归约树不同，所以它是 **BF16 weight-exact**，不是 **stock-token-exact**。若用户要求后者，仍路由 stock BF16。

至此，本地最优不再是一个名字，而是一张有契约的路由表：stock-token-exact 始终走 stock BF16；允许 weight-exact 数值归约差异时，仅 batch 1 走 packed BF16；接受 W4 近似时 batch 1/2 走 W4 scan + BF16 rerank，batch 4 以上走 stock GPTQ-Marlin。机器可读策略见 `models/qwen35_sm89_serving_policy.json`，集成证据见 `comparisons/vllm_bf16_exact_packed_service_integration.json`。这仍是有限候选集上的实测最优，不是全局数学最优。

### 第三十三轮：把“是否快过 vLLM”从人工判断变成 fail-closed 策略

前两轮虽然已经得到分段结论，但策略仍由人手从 JSON 抄写，agent 自己还可能把一次降频的基线、未隔离的编译缓存或第一次服务时发生的 pack/JIT 误认为架构收益。本轮新增公共命令 `scripts/kernel_opt.py service-policy`、输入/输出 schema 和自动推导器。候选只有同时满足以下条件才可晋级：至少两份内容不同的独立 trace；模型、GPU、vLLM、Torch、CUDA、实际 prompt token 哈希和运行参数完全一致；每份正式 trace 的 `VLLM_CACHE_ROOT` 预期值与实测值一致且互不复用；源码哈希和运行时开关可达；预打包与 JIT 在服务前 ready；最后按每个 batch 用“候选各轮最慢值 / 基线各轮最快值”比较。要求 stock-token identity 的契约还会直接比较每个 measured request 的 token IDs。

旧 trace 若缺少后来增加的缓存或 workload 哈希，不能帮助候选晋级；但它若显示更快的库存基线，仍可作为 `BASELINE_ENVELOPE_VETO_ONLY` 抬高门槛。这种非对称证据规则避免 agent 以“证据不够严格”为理由忽略一个对自己不利的强基线。

为闭合输入身份，基准工具新增自身源码 SHA-256、六个实际 prompt-token 序列 SHA-256 和请求轮换规则。随后在本地 4060 上用四个互不相同的冷 `VLLM_CACHE_ROOT` 重跑两轮候选和两轮 stock；候选源码测试完成后已逆向 patch，vLLM 环境恢复原始哈希。

| active batch | stock 正式两轮 | candidate 正式两轮 | 自动保守比值 | 自动路由 |
|---:|---:|---:|---:|---|
| 1 | 102.63 / **103.02** tok/s | 113.89 / **108.05** tok/s | **1.049x** | weight-exact 且允许归约漂移时 candidate |
| 2 | 205.57 / **205.76** tok/s | 206.39 / **193.88** tok/s | 0.942x | stock |
| 4 | 363.71 / 368.52 tok/s | 359.34 / **358.57** tok/s | 0.868x（含更快历史 stock 413.14） | stock |
| 8 | 688.03 / 686.49 tok/s | 696.15 / **655.62** tok/s | 0.941x（含更快历史 stock 696.69） | stock |

因此更严格的回答是：在当前 4060、Qwen3.5-0.8B、greedy、64-token、active batch 1 下，允许 BF16 权重逐 bit 重建但允许 FP32 归约顺序导致 token 漂移时，候选对 stock vLLM 的可保守复现收益是约 **4.9%**，不是翻倍；要求 stock token 完全一致时没有已证明胜出的路径。B2/B4/B8 自动回退 stock。此前的 9.3% 是有效的单轮保守包络结果，但新闭合重复表明它高估了稳定收益，因此部署策略采用 4.9%。这正是新架构要解决的“卡在局部数字、缺少大局观”问题：它产出的是带数值契约和 batch 条件的路由，而不是宣布一个永久最快算子。

机器可读输入为 `models/bf16_serving_policy_spec.json`，自动重算结果为 `models/bf16_serving_policy_auto.json`，严格原始记录为 `traces/vllm_bf16_exact_strict_{stock,candidate}_{a,b}_service_curve.json`。单元测试覆盖数值契约分流、batch 回退、冷缓存隔离、重复不足、生命周期未 ready、workload 不一致，以及“弱基线只可否决不可晋级”。这依然不构成理论最优证明；它只是让 agent 对当前候选集作出更难自欺、可复算的部署决定。

### 第三十四轮：量化 vLLM 也不是单点最优，但胜出必须绑定质量合同

上一轮自动策略只闭合了 BF16。本轮用同一套 fail-closed 规则重新验证 GPTQ：stock 与候选各跑两份独立冷 `VLLM_CACHE_ROOT`，冻结模型、GPU、vLLM/Torch/CUDA、基准脚本 SHA、实际 prompt-token SHA、batch、生成长度和运行开关。候选的输出头 W4 构建也从 `marlin_utils_test` 迁移到正式运行时组件：`gptq_quantize_weights`、`pack_rows`、`gptq_marlin_repack`、`marlin_permute_scales` 和 `marlin_make_workspace_new`；pack 与一次零输入 warmup 均进入 tied embedding 的权重后处理阶段，所以首个服务请求不再承担量化、repack 或冷内核成本。

| active batch | stock GPTQ 正式两轮 | candidate 正式两轮 | 候选最慢 / stock 保守包络最快 | 自动路由（允许有界近似） |
|---:|---:|---:|---:|---|
| 1 | 120.00 / 120.81 tok/s | **177.02 / 179.92 tok/s** | **1.420x**（历史 stock 上限 124.67） | candidate |
| 2 | 274.77 / 258.76 tok/s | **325.35 / 327.91 tok/s** | **1.155x**（历史 stock 上限 281.76） | candidate |
| 4 | 496.17 / 483.27 tok/s | 483.26 / 488.08 tok/s | 0.884x（历史 stock 上限 546.75） | stock |
| 8 | **928.53 / 890.22 tok/s** | 894.01 / 906.63 tok/s | 0.963x | stock |

这组结果回答了“vLLM 量化不就更快吗”：是的，stock GPTQ-Marlin 已经显著快于 stock BF16；但在 active batch 1/2，未量化的超大 tied `lm_head` 仍是残余热点，W4 全词表 shortlist 加 BF16 top-128 回排还能在 vLLM 内部继续取得约 **42.0% / 15.5%** 的保守收益。到 batch 4/8，矩阵路径摊薄权重流量，候选反而落后，所以不存在跨 shape 的单一冠军。

数值合同决定这不是 stock GPTQ 的无条件替代。512 道 GSM8K 中两边都答对 42 道，但只有 457/512 完整 token 序列一致；本轮冻结服务样本的 paired token identity 在 B1/B2/B4/B8 也分别只有 89.1%/96.1%/94.6%/95.0%。因此 `stock_gptq_token_identity` 合同始终自动路由 stock；只有显式接受 `approximate_w4_greedy_quality_bounded` 时，B1/B2 才启用候选。质量 JSON 的 SHA 已作为合同输入绑定，不能只换吞吐 trace 就绕过质量门。

代价同样被显式记录：候选加载占用约 0.99 GiB，stock 约 0.85 GiB，因为当前实现保留 tied dense 权重和 packed 输出头两份表示；加载时间多约三秒。预打包 sidecar 可以删除运行时转换，却不能直接删除 dense 表：输入 embedding 与 BF16 shortlist 回排都还要读取它。若要省掉 dense 表，必须另做量化 embedding/输出头共享格式并重新通过质量门，而不是把它伪装成无语义变化的 checkpoint 优化。机器可读输入和结果分别为 `models/gptq_serving_policy_spec.json`、`models/gptq_serving_policy_auto.json`，原始记录为 `traces/vllm_gptq_prod_{stock,candidate}_{a,b}_service_curve.json`。这证明的是冻结部署点与候选集合中的条件最优，不是理论全局最优。

### 第三十五轮：预打包 sidecar 解决冷启动，不伪装成稳态或显存突破

为了把上一轮候选从“每次启动都临时量化 lm-head”推进到可部署生命周期，本轮增加离线 sidecar 构建器。它读取 checkpoint 中唯一的 tied `embed_tokens.weight`；审计发现该张量物理存储是 FP16，而 vLLM 运行时将其转为 BF16，因此生成器显式复刻 FP16→BF16 后再调用正式的 GPTQ quantize、row pack、Marlin repack 和 scale permutation，分别记录 checkpoint FP16 与运行时 BF16 的 SHA-256。最终 sidecar 为 131,113,688 B，SHA-256 为 `b8a06c4...a1a793`。

等价性测试在 4060 上分别走在线构建与 sidecar 加载：packed weight、permuted scales、空 `g_idx`、排序索引、两行随机输入的 rerank logits 和 argmax 全部逐 bit 相同；伪造 sidecar SHA 会 fail closed。在线 quantize+repack 为 2.047 秒，sidecar 文件哈希校验加 GPU 加载为 0.143 秒，物化阶段快 **14.34x**，减少约 **1.90 秒**。这是启动阶段收益，不应乘到 steady-state tok/s 上。

第一次真实引擎冒烟把 sidecar 放在 Hugging Face 模型目录，vLLM 将目录内所有 `.safetensors` 识别为 checkpoint shards，因额外 tensor 名称失败。该次归类为 `TECHNICAL_FAILURE`，没有否决架构；修复是使用独立 `/home/aden/vllm-sidecars/...` 目录，并让生成器拒绝与 checkpoint 同目录的输出。修复后引擎日志明确报告 sidecar 已加载，模型加载为 2.155 秒、0.98 GiB，单次 discovery B1 为 180.38 tok/s，落在基础候选 177--180 tok/s 的既有范围。

这轮也纠正了一个重要误区：Qwen3.5-0.8B 的 `lm_head` 和输入 embedding 共享同一个 `248320×1024` 权重，且 BF16 top-128 回排同样读取原始行。因此 sidecar 只替代“如何生成 packed 副本”，不能删除 BF16 主表，也不会降低稳态 0.14 GiB 的 packed 额外占用。若只保留 W4 表，就必须同时实现量化 embedding，并放弃现有 BF16 回排权威；那是新的数值候选，已有 W4-only 证据显示行为漂移显著，必须重新资格验证。

所以本轮候选定性为 `DISCOVERY_SMOKE_PASS_LIFECYCLE_HARDENING_ONLY`：它使已有 B1/B2 极速模式更适合频繁冷启动，但不改变自动服务路由、不创造新的稳态吞吐最优，也不声称显存下降。证据见 `candidates/marlin-w4-lmhead-sidecar/summary.json`、`build_manifest.json`、`equivalence_validation.json` 与 `traces/vllm_gptq_sidecar_candidate_smoke_b1.json`。

### 第三十六轮：vLLM 的异步调度已经开启，不能把 kernel 外时间当成白捡收益

旧 profile 分析显示 INT8+BF16 路径加权 TPOT 为 4.260 ms，而 `CUPTI_ACTIVITY_KIND_KERNEL` 总时长除以生成步数仅为 1.193 ms；两次 CUDA Graph launch 起点间隔中位数为 4.249 ms。复核原始 SQLite 后发现恰好另有 186 条 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`，与 186 次 decode graph launch 一一对应，回放主体合计 542.246 ms、中位 2.907 ms。旧脚本漏掉这部分 GPU 工作，因而把约 2.9 ms 错归进了所谓 kernel 外时间。

修正后的分析把 graph replay 与 standalone kernel 一起计入：观测 GPU activity 的保守上界为 4.017 ms/token，占 TPOT 94.29%；剩余下界只有 0.243 ms/token（5.71%）。再对 180 个请求内相邻 GPU graph-start 窗口直接求 activity interval 并集，平均 4.309 ms 窗口中 GPU busy 为 3.967 ms、idle 为 0.342 ms（7.94%）。即使乐观删除观测到的全部平均 idle，推算也只有 1.086x。这里仍不能把 5.71% 或 7.94% 当成可实现收益，因为依赖、submission 与同步不是无条件可删；但足以证明此前错误分解暗示的数倍空间不存在。

本轮用与 benchmark 相同的模型、BF16 dtype 和环境开关只构造 `EngineArgs` 配置，不加载权重。实际配置为：`use_v2_model_runner=False`、`async_scheduling=True`，scheduler 类是 `vllm.v1.core.sched.async_scheduler.AsyncScheduler`，单 GPU `uni` executor 的 `max_concurrent_batches=2`。源码也明确说明 async scheduling 使用两个并行 in-flight batch 来重叠 scheduling/execution。也就是说，已有 vLLM baseline 已经包含这个优化；再做 async-on 只能复现 stock vLLM 的能力，不能成为“快过 stock”的候选。

因此关闭“打开 async scheduling”这个伪机会，也撤回“72% runtime 空隙”这个错误机会。真正有资格重开 runtime orchestration 的候选，必须改变逐 token 依赖链，例如可用的 device-resident 多 token runner，或在真实环境可运行的新 runner；实现前仍需预测至少 3% 的整步收益。当前 V2 runner 在 WSL 下因 UVA 前置条件失败，只能记为环境限制，不能臆测其加速幅度。修正后的机器可读边界与配置审计分别见 `models/vllm_fp8_rerank_decode_cadence_bound.json` 和 `models/vllm_async_scheduling_audit.json`。

## 技术失败与环境边界

- vLLM V2 runner 在当前 WSL 驱动上因 UVA 不可用而失败；固定 `VLLM_USE_V2_MODEL_RUNNER=0` 后兼容 runner 正常。
- FlashInfer top-k/top-p sampler 首次 JIT 需要虚拟环境 `ninja` 在 PATH；本工作负载是 greedy，固定 `VLLM_USE_FLASHINFER_SAMPLER=0`，避免无关采样器污染主干验证。
- 当前虚拟环境同时存在 CUDA 13.0 runtime headers、CUDA 13.3 `nvcc`，系统还有 CUDA 12.0 `nvcc`。需要 JIT 的候选必须先做 compiler/header 一致性 preflight；本轮 FP8 KV cache 因此只记技术失败，不作性能拒绝。
- 当前 Nsight Compute counters 受 `ERR_NVGPUCTRPERM` 限制，所以没有伪造 cache/issue/stall 归因。
- WSL 系统自带的 Nsight Systems 2022.4 无法导入 CUDA 13 profiler-range capture；后来在用户目录安装 2025.5.1 后已经成功获得 CUDA Graph node timeline。旧失败文件仍只作技术记录，新 SQLite timeline 用于机会排序和运行时可达性证明。
- 5090 未被访问或占用，遵守其正在运行 MiniMax-H3 的约束。
- `lm_head` 候选已完成带功耗/温度遥测的 3×10、同源 C-S-S-C 和 nsys GPU-active 拆分，但仍不是最终 SOTA certificate；尚未完成锁定电源模式后的随机进程交错、更大质量集和最终 binary/SASS 审计。

## 复现入口

所有脚本、原始样本和推导都位于本 run：

- `tools/smoke_transformers_reference.py`
- `tools/profile_transformers_reference.py`
- `tools/smoke_vllm_offline.py`
- `tools/benchmark_vllm_offline.py`
- `tools/validate_vllm_against_transformers.py`
- `tools/benchmark_memory_stream.py`
- `tools/inventory_model_snapshot.py`
- `tools/analyze_bandwidth_bound.py`
- `traces/vllm_discovery_baseline_w1_n3.json`
- `traces/vllm_vs_transformers_parity.json`
- `traces/sm89_memory_stream_model_sized.json`
- `models/bandwidth_bound.json`
- `models/feasibility_gate.json`
- `models/vllm_candidate_search.json`
- `traces/vllm_confirm_fastloop_cuda_a_w1_n7.json`
- `traces/vllm_confirm_fastloop_triton_w1_n7.json`
- `traces/vllm_confirm_maxseq80_cache512m_repeat_w1_n7.json`
- `tools/summarize_vllm_search.py`
- `tools/benchmark_bf16_skinny_gemm.py`
- `tools/compare_vllm_candidate.py`
- `patches/vllm_0.28.1_sm89_bf16_lm_head.patch`
- `models/sm89_lm_head_candidate.json`
- `models/sm89_lm_head_comparison.json`
- `models/sm89_lm_head_abba.json`
- `models/sm89_selective_backbone_ablations.json`
- `models/sm89_nsys_operator_map.json`
- `models/sm89_combined_bf16_frontier.json`
- `models/sm89_strict_bf16_residual_certificate.json`
- `models/sm89_mtp_m2_projection_bound.json`
- `models/sm89_recurrent_state_search_bound.json`
- `models/sm89_recurrent_norm_fusion_stop.json`
- `models/nsys2025_gdn_segmented_candidate_map.json`
- `models/nsys2025_lmhead_opportunity_map.json`
- `models/nsys2025_reachable_gdnqkvz_map.json`
- `models/nsys2025_reachable_all_map.json`
- `models/nsys_tool_identity.json`
- `models/opportunity_map.json`
- `models/qwen35_projection_dataflow_audit.json`
- `models/opportunity_specs/*.json`
- `candidates/gdn-qkvz-ba-fusion/screening_summary.json`
- `candidates/gdn-segmented-projection/summary.json`
- `candidates/gdn-segmented-projection/vllm_qwen_gdn_segmented_projection.patch`
- `candidates/reachable-backbone-gemv/summary.json`
- `candidates/reachable-backbone-gemv/vllm_utils_reachable_backbone_gemv.patch`
- `candidates/fused-lmhead-argmax/summary.json`
- `candidates/combined-gdn-down/summary.json`
- `candidates/exact-vocab-pruning/summary.json`
- `candidates/exact-bf16-packed-lmhead/summary.json`
- `candidates/fp8-block-exact-lmhead/summary.json`
- `candidates/vllm-online-quantization-sm89/summary.json`
- `candidates/int8-bf16-shortlist-rerank/summary.json`
- `comparisons/vllm_fp8_exact_vs_rerank_g1024k128_qualification.json`
- `models/nsys2025_int8_bf16_rerank_g1024k128_map.json`
- `comparisons/vllm_fp8_native_vs_exact_qualification.json`
- `candidates/exact-bf16-packed-lmhead-small-batch/summary.json`
- `candidates/exact-bf16-packed-lmhead/vllm_utils_exact_bf16_packed_lmhead.patch`
- `candidates/exact-bf16-packed-backbone/summary.json`
- `candidates/exact-lmhead-segmented-gdn/summary.json`
- `microbench_candidates/fused_lmhead_argmax_sm89.json`
- `microbench_candidates/exact_bf16_packed_lmhead_sm89_steady.json`
- `microbench_candidates/exact_bf16_codebook_lmhead_sm89.json`
- `microbench_candidates/exact_bf16_backbone_cold_stream_sm89.json`
- `microbench_candidates/exact_bf16_backbone_tensorcore_attention_sm89.json`
- `tools/benchmark_sm89_fused_lmhead_argmax.py`
- `tools/analyze_exact_bf16_weight_compression.py`
- `tools/benchmark_exact_bf16_packed_lmhead.py`
- `tools/benchmark_exact_bf16_codebook_lmhead.py`
- `tools/analyze_exact_bf16_backbone_compression.py`
- `tools/benchmark_exact_bf16_backbone_stream.py`
- `tools/benchmark_exact_bf16_backbone_tensorcore.py`
- `tools/benchmark_marlin_int4_recall_lmhead.py`
- `microbench_candidates/marlin_int4_recall_lmhead_m2_sm89.json`
- `comparisons/vllm_fp8_marlin_w4_rerank_mtp1_screen.json`
- `comparisons/vllm_gptq_marlin_w4_stock_vs_specialized_head_low_power.json`
- `tools/benchmark_vllm_batch_service.py`
- `comparisons/vllm_bf16_gptq_rerank_service_frontier.json`
- `traces/vllm_bf16_stock_service_curve_b1_b2_b4_b8.json`
- `traces/vllm_gptq_stock_service_curve_b1_b2_b4_b8.json`
- `traces/vllm_gptq_stock_service_curve_repeat_same_cache_b1_b2_b4_b8.json`
- `traces/vllm_gptq_mv_rerank_service_curve_same_cache_b1_b2_b4_b8.json`
- `comparisons/vllm_bf16_exact_packed_service_integration.json`
- `models/qwen35_sm89_serving_policy.json`
- `models/bf16_serving_policy_spec.json`
- `models/bf16_serving_policy_auto.json`
- `models/gptq_serving_policy_spec.json`
- `models/gptq_serving_policy_auto.json`
- `models/vllm_async_scheduling_audit.json`
- `traces/vllm_bf16_exact_strict_stock_a_service_curve.json`
- `traces/vllm_bf16_exact_strict_stock_b_service_curve.json`
- `traces/vllm_bf16_exact_strict_candidate_a_service_curve.json`
- `traces/vllm_bf16_exact_strict_candidate_b_service_curve.json`
- `traces/vllm_bf16_exact_load_warm_service_curve_b1_b2_b4_b8.json`
- `traces/vllm_bf16_exact_load_warm_stock_close_service_curve_b1_b2_b4_b8.json`
- `traces/vllm_gptq_prod_stock_a_service_curve.json`
- `traces/vllm_gptq_prod_stock_b_service_curve.json`
- `traces/vllm_gptq_prod_candidate_a_service_curve.json`
- `traces/vllm_gptq_prod_candidate_b_service_curve.json`
- `traces/vllm_gptq_marlin_w4_stock_natural_128_low_power.json`
- `traces/vllm_gptq_marlin_w4_plus_w4_head_rerank_natural_128_low_power.json`
- `traces/vllm_gptq_marlin_w4_plus_w4_head_rerank_k512_screen.json`
- `candidates/gptq-marlin-backbone-fair-baseline/quantization_config_self_attn_fix.patch`
- `models/sm89_exact_vocab_pruning_feasibility.json`
- `tools/analyze_exact_vocab_pruning.py`
- `traces/vllm_reachable_downw2_64.json`
- `traces/vllm_reachable_downw8_64.json`
- `comparisons/vllm_sm89_mlp_quality_gsm8k_n512.json`
- `profiles/nsys2025_lmhead_candidate_nodes.sqlite`
- `profiles/nsys2025_reachable_gdnqkvz_b.sqlite`
- `profiles/nsys2025_reachable_all_b.sqlite`
- `comparisons/vllm_lmhead_toggle_pair1.json`
- `comparisons/vllm_lmhead_toggle_pair2.json`
- `comparisons/vllm_exact_pack_qualification.json`
- `comparisons/vllm_exact_pack_stock_qualification.json`
- `comparisons/vllm_exact_gdn_qualification.json`
- `comparisons/vllm_sm89_gdn_quality_gsm8k_n512.json`
- `microbench_candidates/bf16_triton_mtp_m2_backbone_sm89.json`
- `microbench_candidates/gdn_packed_decode_bf16_state_sm89.json`
- `microbench_candidates/gdn_packed_decode_bf16_state_multiwarp_sm89.json`
- `microbench_candidates/gdn_recurrent_norm_fusion_sm89.json`
- `tools/analyze_mtp_m2_projection_bound.py`
- `tools/benchmark_vllm_gsm8k_quality.py`
- `tools/build_marlin_lmhead_sidecar.py`
- `tools/validate_marlin_lmhead_sidecar.py`
- `candidates/marlin-w4-lmhead-sidecar/summary.json`
- `candidates/marlin-w4-lmhead-sidecar/build_manifest.json`
- `candidates/marlin-w4-lmhead-sidecar/equivalence_validation.json`
- `candidates/marlin-w4-lmhead-sidecar/vllm_utils_sidecar_loader.patch`
- `traces/vllm_gptq_sidecar_candidate_smoke_b1.json`
- `tools/benchmark_gdn_recurrent_norm_fusion.py`
- `tools/compare_gsm8k_quality.py`
- `traces/vllm_natural_sm89_lmhead_gemv_qual_n_w3_n10.json`
- `traces/vllm_natural_stock_qual_o_w3_n10.json`
- `traces/vllm_lmhead_restored_smoke_w1_n1_t16.json`
- `models/sm89_exact_bf16_backbone_compression_feasibility.json`
- `models/nsys2025_exact_packed_frontier_map.json`
- `profiles/nsys2025_exact_packed_frontier_nodes.sqlite`
- `traces/vllm_nsys2025_exact_packed_frontier_w1_n1_t32.json`
- `traces/vllm_exact_gdn_control_w3_n10.json`
- `traces/vllm_exact_gdn_candidate_w3_n10.json`

ModelScope 模型页：<https://modelscope.cn/models/Qwen/Qwen3.5-0.8B>
