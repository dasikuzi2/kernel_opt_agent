# vLLM 通用适配设计

## 结论

这个候选不是把 Qwen3.5-0.8B 的形状写死进 vLLM，而是增加一个独立的
`lm_head_backend`。它对任意 `ParallelLMHead(N, K)` 做能力判断、无损布局规划和
运行时路由；不满足条件时返回原有 vLLM 路径。默认值仍为 `torch`，实验后端必须
显式选择，避免把一张卡、一个模型的局部胜利推广成全平台默认值。

## 为什么接在这里

vLLM 的输出链路是：模型完成权重加载与可能的 embedding 重绑，然后调用量化方法的
`process_weights_after_loading`，推理时由 `LogitsProcessor._apply_head` 调用
`lm_head.quant_method.apply`，最后继续执行 TP gather、裁掉 padding、缩放和采样。

适配因此分为四个很小的边界：

1. `KernelConfig` 增加 lm-head 专用后端和最大辅助布局比例；不挤占量化 linear backend。
2. 在权重重绑完成后的正式预处理钩子里建立无损辅助布局。
3. 只在 `_apply_head` 的投影位置尝试 M=1 快路径，其他语义仍由原实现拥有。
4. 使用 vLLM 的共享 `VllmJitKernel` 注册表在 CUDA Graph 捕获前预热编译键。

## 通用性边界

静态支持不是“所有模型都一定更快”。当前能力谓词要求 CUDA、BF16、二维连续权重、
`K <= 8192`、有符号 32 位索引安全且 dense head 至少 64 MiB。布局超过 dense BF16
的 90%、预留显存不足，或运行时不是无 bias 的连续 M=1 BF16 输入时，自动回退。
N、K 和 TP 后的本地词表分片都不是硬编码值。

这种分层刻意区分：

- **可运行性**由 vLLM 内部能力谓词决定；
- **是否值得启用**由 agent 在目标模型、GPU 和真实 workload 上做成对端到端 A/B 决定；
- **是否能声称最优**仍需完整候选集、官方硬件证据、最终二进制和置信区间，不能由一次 A/B 得出。

## 数值与内存

每个 256-value block 保存 1 byte sign+mantissa、两个 4-bit exponent delta、base exponent，
指数跨度超过 15 的 block 存 exact BF16 fallback。原 BF16 weight 不删除，因此任何不支持
的请求都能走原路径。打包分两遍、按 chunk 执行，避免原型为整张矩阵制造数 GiB 的
临时 int32 tensor；在分配前还保留 256 MiB workspace headroom。

“无损”只指 weight bit reconstruction。Triton 与 cuBLAS 的 FP32 reduction tree 不同，
logits 不保证 bitwise identical。对普通容差语义这是常见的合法差异；对跨进程 token
严格一致的实验，必须先证明 stock 本身确定，再单独设立 deterministic contract。

## 面向 agent 的最终形态

生产后端只负责安全能力判断和回退；agent 负责生成经过目标 workload 验证的配置：

```json
{
  "kernel_config": {
    "linear_backend": "marlin",
    "lm_head_backend": "lossless_packed",
    "lm_head_max_packed_fraction": 0.9
  }
}
```

不建议在模型加载时用孤立 microbenchmark 自动决定默认值。此前两个原子 shape 的 Humming
测试虽然局部获胜，组合到全模型反而变慢；最终路由必须由端到端目标函数裁决。
