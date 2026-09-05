# 通用 vLLM 算子适配与 RTX 4060 验证报告

## 当前结论

本轮把原先写死 Qwen3.5-0.8B、`N=248320/K=1024`、SM89 的 BF16 output-head
原型改成了 vLLM 内的 shape-generic 实验后端。实现不识别模型名，也不写死词表和 hidden
size；它识别 vLLM 的 `ParallelLMHead`、dtype、布局、运行 shape、辅助内存和压缩收益，
不满足条件就回到原有 vLLM 路径。

在本机 RTX 4060 Laptop GPU、Qwen3.5-0.8B GPTQ-Marlin backbone、BF16 activation 与
BF16 tied lm_head、batch 1、每请求生成 64 token 的 6 个自然 prompt 上，最终源码哈希匹配的
成对结果是：

| 指标 | stock vLLM | 通用后端 | 变化 |
|---|---:|---:|---:|
| 平均端到端延迟 | 344.440 ms | 304.207 ms | -11.681%，1.1323x |
| 平均 TPOT | 4.587 ms | 3.967 ms | -13.50% |
| 平均 TTFT | 56.525 ms | 52.188 ms | -7.67% |
| 输出吞吐 | 218.04 tok/s | 252.19 tok/s | +15.66% |
| 测量样本 | 6 case × 3 | 6 case × 3 | 对称 |
| 测量后 graphics clock | 2625 MHz | 2625 MHz | 相同 |
| 生成 token | 6/6 case | 6/6 case | 与成对 stock、canonical stock 一致 |

这是 discovery evidence，不是 production acceptance，也不是理论最优证明。当前结果的 baseline
是 GPTQ-Marlin 模型中的 stock BF16 output head，并非整个 BF16 模型与整个量化模型的等价比较。

## vLLM 架构与接入位置

当前 vLLM 把量化 linear kernel 的选择放在 `KernelConfig.linear_backend`，模型层通过
quantization method 的 `process_weights_after_loading` 完成 load-time 转换；输出投影由
`LogitsProcessor._apply_head` 调用 lm-head quant method，之后继续执行 tensor-parallel gather、
去 padding、logits scale 与 sampling。vLLM 还提供共享的 `VllmJitKernel` warmup registry，
用于在 CUDA Graph capture 之前编译实际 shape。

本实现采用如下链路：

```text
KernelConfig.lm_head_backend
          │
          ▼
process_weights_after_loading ──不满足能力谓词──► stock weight/method
          │
          ▼
lossless auxiliary layout + JIT warmup
          │
          ▼
LogitsProcessor._apply_head ──M!=1/bias/类型不符──► stock method
          │
          ▼
existing TP gather / trim / scale / sampling
```

没有复用 `linear_backend` 是有意的：它目前表示量化 GEMM 后端，而 BF16 lm_head 有 embedding
重绑、vocab TP shard 和 M=1 decode 等不同生命周期。独立配置项能避免污染所有 linear layer。

参考的上游接口：

- vLLM `KernelConfig`: https://docs.vllm.ai/en/latest/api/vllm/config/kernel/
- vLLM linear kernel API: https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/
- `UnquantizedEmbeddingMethod` / `ParallelLMHead`: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
- `LogitsProcessor`: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py

## 具体修改

1. 新增 shape-generic Triton 后端：任意安全的 `N/K`、任意 TP 后本地 vocab shard，不含模型名判断。
2. 新增 `lm_head_backend=torch|lossless_packed` 与 `lm_head_max_packed_fraction`；默认 `torch`。
3. 在正式 load-time hook 中打包，接入共享 JIT warmup，确认真实运行日志在 CUDA Graph 前出现
   `_PackedBF16LMHeadKernel (1 keys)`。
4. 将打包改成 two-pass chunked 算法，避免旧原型对整张 weight 建立数 GiB int32 临时张量。
5. 增加 64 MiB materiality、`K<=8192`、signed-32-bit index、packed ratio 和 256 MiB
   workspace headroom 检查。
6. 保留原 BF16 weight。M>1、bias、非 BF16、非连续输入或任何不支持场景都走 stock 方法。
7. benchmark harness 新增 lm-head 后端参数、contract hash binding、源文件 hash guard，并把错误的
   contract 路径检查前置到 engine 初始化之前。
8. 新增纯 Python bit reconstruction 检查、非 2 次幂 CUDA shape 测试、成对证据验证器和候选说明。

主要文件：

- `candidates/adaptive-lossless-lmhead/vllm/model_executor/kernels/linear/unquantized/lossless_packed_lm_head.py`
- `candidates/adaptive-lossless-lmhead/vllm_integration.patch`
- `candidates/adaptive-lossless-lmhead/validate_evidence.py`
- `candidates/adaptive-lossless-lmhead/gpu_shape_smoke.py`
- `candidates/adaptive-lossless-lmhead/smoke_result.json`
- `raw/stock_marlin_power_matched_r43.json`
- `candidates/adaptive-lossless-lmhead/raw_r44.json`

## 通用 shape 验证

独立于 Qwen 的 CUDA case 使用 `[N,K]=[33001,1023]`，刻意覆盖非 2 次幂 K、非整 block 尾部和
exact fallback：

- sampled BF16 weight bits: 8192/8192 exact；
- packed fraction: 0.7653；fallback blocks: 724；
- argmax 与 `torch.nn.functional.linear` 相同；
- 最大 logit 差 0.125，为 logit 峰值的 0.1724%；
- M=2 与带 bias 路径都正确回退。

“lossless”严格指 weight bit reconstruction。Triton reduction tree 与本次 cuBLAS algorithm
不同，因此不能承诺 logits bitwise identical。更重要的是，本轮 stock 在不同冷启动之间自己也有
2/6 case 的 greedy token 漂移；候选的三轮输出反而与 canonical stock 6/6 一致。所以后续严格
正确性门应同时测 stock-stock stability、logit tolerance、top-1 margin 与任务指标，不能把一次
跨进程 token 串当数学证明。

## 为什么这比原型更接近“通用 agent”

原型回答的是“这个固定 shape 能不能快”；新结构把三个责任拆开：

- vLLM backend 负责能力判断、权重生命周期、编译预热和安全回退；
- agent 负责在目标 GPU、模型、batch 分布和服务目标上生成候选并做完整端到端 A/B；
- global scheduler 负责比较候选组合，拒绝局部 microbench 胜利但全模型退化的方案。

本轮已有直接反例支持这种分工：两个 Humming shape 在原子测试中获胜，组合进模型后仍比 stock
慢 3.99%。因此不应在 vLLM 模型加载时根据一个孤立 microbenchmark 自动宣布 backend；应由
agent 生成经过 workload 验证的 `KernelConfig`。

## 尚未解决与下一步

- 目前快路径只覆盖 M=1，batch 2/4/8 会回退，因此还不是“所有场景统一加速”。下一候选应是
  M=2/4/8 的 GEMM/skinny-GEMM 家族，而不是继续微调 M=1 block size。
- 额外辅助布局为 375.519 MiB，即 dense BF16 head 的 0.774；这是用显存换带宽，显存紧张时会回退。
- 5090 尚未验证，因为该机器按用户要求留给 MiniMax-H3；不能把 4060 结果外推成 Blackwell 结果。
- 当前 run 的 discovery 总 wall-clock budget 已被先前实验耗尽。新候选已登记为 `PROPOSED`，
  手工 build/correctness/smoke 均已执行，但框架正确拒绝了新的 candidate-run。应在新 run 中用新的
  20 分钟候选预算复现，而不是篡改旧预算。
- 还没有官方硬件资料闭环、最终 SASS/resource audit、置信区间和 6--12 个多家族候选组合，因此
  不能声称理论最优或 SOTA。当前可以声称的是：通用接入可运行，4060 上这个 workload 有稳定的
  discovery-level 约 11.7% 端到端改进。
