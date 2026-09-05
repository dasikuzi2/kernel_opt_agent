# 用 RTX 4060 判断该继续调核，还是应该换架构

这份笔记不是 CUDA/Triton 语法教程，而是一份可以复用的性能研究方法。目标是回答：当一个 kernel 已经变快后，时间究竟还花在哪里，以及下一单位开发时间应该投入哪种改动。

## 一、先冻结能回答的问题

本次 4060 是 SM89，而目标 GDN 主体只支持 SM90/SM100/SM120。4060 因此只能回答：

- gate 公式能否正确融合；
- 自动编译与手写单核的 launch 拓扑差异；
- 动态 shape 特化是否拖慢 agent 搜索；
- producer-consumer 直接融合是否存在可迁移的机制收益。

它不能回答完整 GDN 在 5090 上的绝对延迟、最佳 tile、occupancy、SOTA 差距或理论最优距离。先写清边界，是为了避免“数字很漂亮，问题却答错了”。

## 二、把一个延迟拆成三个层次

同一个操作至少要分别观察：

| 层次 | 本次计时方法 | 它回答什么 | 不能直接当成什么 |
|---|---|---|---|
| kernel active | CUPTI CUDA activity | GPU 真正在执行该 kernel 的活动区间 | 完整调用延迟 |
| effective GPU timeline | 一批 CPU 串行 launch 外围的 CUDA events | 应用按当前方式连续发射时，GPU 时间线上每次调用的平均间隔 | 纯 kernel duration 或纯 launch overhead |
| synchronized host wall | 同步前后 `perf_counter` | 调用者实际等待的墙钟成本 | GPU 硬件能力 |

CUDA events 包住大量 Python launch 时会包含 GPU 空闲间隙，所以不能把 `effective - active` 全叫作 CUDA launch overhead。本次保留这个差值，但把它命名为 `effective_minus_active`。

## 三、成本栈实验告诉了什么

固定 6 个形状、`BLOCK=256`、9 轮 AB/BA、每轮 500 次，并在 3 个独立进程中完整重复：

- 预分配输出的跨进程中位平均有效时间 `19.93us`，范围 `19.70～23.65us`；
- 每次重新分配输出为 `34.65us`，范围 `34.35～40.61us`；
- 输出分配/包装路径差值中位数 `14.95us`；
- CUPTI gate kernel active 中位数 `1.464us`，范围仅 `1.346～1.468us`；
- kernel active 占有效时间的跨进程中位数约 `6.75%`。

这时继续优化 exp、softplus 或 warp 数，即使把 kernel active 不现实地减半，也只影响约 `0.73us`。相反，消除分配、launch 或整段中间张量可以作用于十几微秒。这个数量级比较就是“大局观”的可执行版本。三个进程中 active 很稳定而 effective 明显漂移，也揭示了 WDDM/display 调度噪声；因此这里晋级因果机制，不签发精密硬件极限。

原始数据：`traces/gate_cost_stack_sm89_proxy_rep1.json` 至 `rep3.json`，跨进程汇总为 `traces/sm89_proxy_repetition_summary.json`。

## 四、如何做 producer-consumer 因果实验

两条路径保持输入、输出、dtype、元素数、grid 和 `BLOCK=256` 相同，只改变一件事：

```text
物化路径：gate kernel -> 写 g/beta -> consumer kernel 读 g/beta -> out
融合路径：读取 gate 输入 -> 现场计算 g/beta -> 立即参与 consumer -> out
```

synthetic consumer 固定为：

```text
out = exp(-exp(A_log) * softplus(a + dt_bias)) * x + sigmoid(b) * y
```

结果：

| 指标 | 两核物化 | 一核直接消费 | 加速 |
|---|---:|---:|---:|
| 跨进程中位平均有效时间 | 37.77us | 20.44us | 1.87x |
| 跨进程中位平均 CUPTI active | 2.776us | 1.591us | 1.79x |

三次独立进程中 6/6 形状都通过 `atol=rtol=1e-5`，每个形状方向一致；effective 加速范围 `1.84～1.91x`，active 加速范围 `1.77～1.79x`。每元素在逻辑上删除了两份 FP32 中间量的一次写和一次读，即 `16 bytes/element`；这里只能称为逻辑流量，缓存可能吸收其中一部分，不能在没有计数器时称为 DRAM 流量。

为什么这个对照比继续扫参数有价值：它同时在 effective timeline 和 kernel active 两个相互独立的观察层上超过预设 `1.05x` 门槛，指向同一个因果机制。相反，上一轮按形状挑 BLOCK/warp 的 oracle 收益只有 `2.08%`。

原始数据：`traces/gate_consumer_fusion_sm89_proxy_rep1.json` 至 `rep3.json`。

## 五、以后自己判断的四步规则

1. 先画数据流：有几个 launch，哪些中间量被写回又读回。
2. 分别量 active、effective 和 host wall，不把一个数字解释成所有层次。
3. 算收益上限：若目标局部只占总时间 5%，局部快 2 倍也只能带来约 2.5% 全局收益。
4. 设停止线：本次规定 shape dispatch 的潜在收益 `<2%` 直接拒绝、`2%～5%` 延迟到目标机、`>5%` 才值得增加分派复杂度。

一个实用判断是：当 kernel active 只占 effective timeline 很小一部分时，优先考虑批处理、图捕获、scratch 复用、producer-consumer 融合和把预处理并入主体；只有 active 占比高且计数器指出具体资源瓶颈时，才进入指令、tile、warp 和流水线调优。

## 六、如何复现

在 Windows PowerShell 中从仓库根目录执行：

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/codes/kernel_opt_agent && /home/aden/.venvs/kernel-opt-4060/bin/python runs/20260901_flashinfer_gdn_prefill_sm120_v1/tools/profile_gate_cost_stack_sm89.py --candidate runs/20260901_flashinfer_gdn_prefill_sm120_v1/candidates/gate_fusion_auto/main.py --screening-set runs/20260901_flashinfer_gdn_prefill_sm120_v1/models/architecture_screening_set.json --output runs/20260901_flashinfer_gdn_prefill_sm120_v1/traces/gate_cost_stack_sm89_proxy_rep1.json'
```

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/codes/kernel_opt_agent && /home/aden/.venvs/kernel-opt-4060/bin/python runs/20260901_flashinfer_gdn_prefill_sm120_v1/tools/benchmark_gate_consumer_fusion_sm89.py --candidate runs/20260901_flashinfer_gdn_prefill_sm120_v1/candidates/gate_fusion_auto/main.py --screening-set runs/20260901_flashinfer_gdn_prefill_sm120_v1/models/architecture_screening_set.json --output runs/20260901_flashinfer_gdn_prefill_sm120_v1/traces/gate_consumer_fusion_sm89_proxy_rep1.json'
```

脚本会保存全部 trial 和 CUPTI active 样本，不只保存中位数。

## 七、为什么现在还没有 occupancy/SFU/缓存结论

Nsight Compute 2022.4.1 已连接到正确进程和正确 kernel，但驱动返回 `ERR_NVGPUCTRPERM`。原始输出保存在 `traces/gate_sm89_ncu_basic.csv`，结构化凭据在 `traces/gate_sm89_ncu_access_receipt.json`。

根据 NVIDIA 官方说明，Windows 上需要管理员在 NVIDIA App 的 `System > Advanced > Developer > Manage GPU Performance Counters` 中授权；旧界面也可通过 NVIDIA Control Panel 的 Developer 设置完成。修改后应先复跑同一个 NCU 命令，确认计数器可用，再考虑更新较旧的 Nsight 版本。未经授权时，不应根据 kernel 时间反推 occupancy、SFU 利用率或 DRAM/L2 带宽。

## 八、这轮之后的工程决策

4060 已经足以否决“继续深挖 BLOCK/warp”并晋级“把 gate 直接融合进 recurrent consumer”的机制。它还没有、也无法完成完整 SM120 算子的资格验证。

5090 空闲后的正确顺序是：

1. 先跑 FlashInfer 完整 100-case 独占基线；
2. 在 6-case 上实现 gate-in-recurrent 候选，而不是只保留前置 gate kernel；
3. 验证寄存器压力、occupancy 和 recurrent dependency 是否抵消融合收益；
4. 幸存者回到 100-case，和固定版本 FlashInfer 做同机交错比较；
5. 最后才谈 SOTA 差距与可达上界。
