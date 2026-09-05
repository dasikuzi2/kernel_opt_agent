# Kernel Opt Agent：SM120 GDN 实践与验证报告

更新时间：2026-09-02
分支：`feature/opportunity-driven-search`

## 结论先行

本分支把原先容易长期停留在建模、测量和局部微调的流程，改造成“先确认全局机会，再限时生产候选，最后才把少数幸存者送进重型验证”的闭环。它不能数学证明未知实现空间中的绝对最优，也不能保证一次运行就超过所有公开 SOTA；它能做到的是：更早产出可运行实现、显式淘汰低价值死路、用完整 workload 而非单点结果约束判断，并为“离已知最优还有多远”保留可审计证据。

5090 实践已经证明环境、官方数据、SM120 路径和候选打包链路可以独立搭建。最小 workload 在严格 `atol=rtol=0.01`、100% 元素匹配下通过；当前仍需在 GPU 独占窗口完成 100-case 正式基线及候选比较，才能给出可信的全局加速数字。

## 架构改动

### 1. 从“直接试优化”改为“机会地图驱动”

新增 `opportunity_map.py`，要求每个机会绑定当前全局目标中的可见耗时、可移除上限、可能收益区间、置信度、实现成本和证据哈希。排序使用“预期全局收益 / 实现分钟”，并禁止把条件分解模型包装成绝对最优证明。

这解决了 agent 容易纠缠低占比细节的问题：候选必须回答它减少的是哪一段全局成本，收益上限低于其他机会时会自然后排。

### 2. 增加有硬预算的生产候选发现通道

新增 `candidate_discovery.py`。默认要求 6～12 个候选、至少 4 个改写家族，单候选最多 20 分钟、整个发现组合最多 2 小时、单候选最多 8 次技术修复。候选先经历 build、correctness、anchor/edge smoke；只有少数幸存者才进入完整资格验证。

这把“几十小时还在测量、没有算子产物”改成失败封闭的状态机：没有工作实现时，下一步只能实现当前最高价值候选；技术失败可在预算内修复，但不能无限重试；组合没有筛完时不能偷偷晋级某一个偶然快的候选。

### 3. 引入方法库，但不让外部信息替代实测

新增 `method_library.py` 和带来源、适用前提、失效模式的 method cards。公开论文或实现只作为候选生成先验，不能提高测得收益、不能证明硬件能力，也不能直接通过正确性或性能门槛。

因此“自我迭代”和“外部学习”不是二选一：外部资料负责扩大架构假设空间，目标 GPU 上的冻结 workload 负责裁决。最终实现仍可要求自包含；借鉴方法不等于直接调用外部算子。

### 4. 接入真实 FlashInfer Trace，而不是自造单例

新增 `trace-intake`：冻结定义、参考实现哈希、100 个等权 workload、所有 tensor blob 身份、正确性阈值和官方聚合语义。本次目标是 `gdn_prefill_qk4_v8_d128_k_last`，序列总长 6～8192、batch 1～57，覆盖短序列、长序列和可变长边界。

### 5. 增加快筛子集，而不牺牲最终全量验证

新增 `select_screening_cases.py` 与 `materialize_flashinfer_subset.py`。筛选器按 SM120 context-parallel 路由边界，把两侧分别取最短、中位、最长，共 6 例；tensor blob 用符号链接复用，不复制约 1.1GB 数据。该子集只允许 `DISCOVERY_ONLY` 声明，最终接受仍必须回到 100-case 全量。

这让一次架构候选的早期否决从约二十分钟降到分钟级，同时避免只用一个“小而好看”的 case。

## 5090 环境与实测发现

- GPU：RTX 5090，SM120，170 SM，约 32GB 显存。
- 隔离环境：Python 3.12.3、PyTorch 2.13.0+cu130、Triton 3.7.1、FlashInfer 0.6.18、CUTLASS DSL 4.7.1。
- 官方 B200/SM100 baseline 不能直接迁移：其 `tcgen05` MMA 只接受 SM100/103/110，在 SM120a 编译立即失败。流程只尝试一次便标记 `TECHNICALLY_BLOCKED`，没有浪费时间修补错误架构。
- 第一版 wrapper 把 `log_g` 传给 SM120 AlphaProcessor，导致全量 NaN。检查实际 ABI 后改为正值 `g=exp(log_g)`；最小例全部元素正确，最大绝对误差约 `9.67e-4`。
- 一次全量跑在 69 例通过后出现 7 个 runtime error。检查时发现 ComfyUI 在运行中重新占用约 26.8GB 显存，因此该批次已经标记为 `INVALID_COMPETING_LOAD`，没有被当作候选缺陷或性能证据。
- 本地 RTX 4060 Laptop 是 SM89；当前 FlashInfer GDN 实现只路由 SM90/100/120，因此不能冒充目标算子性能环境。它仍可用于框架测试、数据准备和架构无关的阶段代理测试。
- 4060 上已直接加载 `gate_fusion_auto/main.py` 中的同一 Triton gate kernel：6/6 筛选形状和一组宽值域压力输入均通过 `atol=rtol=1e-5`；PyTorch eager 路径每次 9 个 CUDA kernel，融合路径每次 1 个；交错 AB/BA 配对测得 gate 阶段形状平均约 3.30 倍加速。该数字只证明融合机制值得保留，不能外推为完整 GDN 或 5090 加速。

### 4060 第二轮：编译器、手写核与搜索成本

这一轮预先冻结为两个问题，完成固定矩阵后停止，没有继续追逐单点最优：

1. 先在短、长两个形状筛选 `torch.compile` 模式：`reduce-overhead` 因 CUDA Graph 双输出拷贝约为 `149～157us`，`default` 与 `max-autotune-no-cudagraphs` 约为 `98～99us`，因此正式 6-case 比较采用更强的 `default`，而不是拿较弱模式当对手。正式结果中的 eager、最强自动编译与手写 Triton 均通过 `1e-5` 正确性；形状平均延迟约为 `121.17us`、`97.99us`、`36.05us`，手写路径仍比最强自动编译快约 `2.72x`。profiler 显示 eager 为 9 个 CUDA kernel，自动编译降为 2 个，手写路径为 1 个。说明自动编译已经消掉大部分中间算术，但两个独立输出仍没有合并进同一 program，手写融合有真实机制价值。
2. 固定扫描 `BLOCK={64,128,256,512}` × `num_warps={1,2,4}`。不同形状的单点冠军不稳定；本轮全局冠军为 `b256_w2`，但“每形状选冠军”相对一个全局配置的平均潜在收益只有 `2.08%`，落入预先规定的 `2%～5%` 不确定区间。因此不在 SM89 上增加 shape dispatch，也不把代理机冠军搬到 SM120；目标机再测。
3. 原候选把 `n_elements` 声明为 `tl.constexpr`，6 个冻结形状产生 5 个进程内编译 specialization。仅删除注解并不够，因为 Triton 仍会隐式按值/对齐特化；同时使用 `do_not_specialize` 和 `do_not_specialize_on_alignment` 后降为 1 个 specialization。配对稳态测量只慢约 `1.13%`，因此新增 `gate_fusion_runtime_extent_auto` 候选，保留原 `BLOCK=256`，只迁移“减少无必要编译分叉”这一架构无关结论。

这组实验把两类成本分开了：GPU 稳态延迟决定候选是否值得保留，编译 specialization 数和编译路径墙钟决定 agent 每小时能完成多少次有效假设。对应经验已沉淀为通用 method card `triton-dynamic-extent-specialization-control`，后续 agent 不应再通过无限枚举动态长度来消耗搜索预算。

### 4060 第三轮：成本栈与 producer-consumer 直接融合

进一步把同一个 gate kernel 拆成三个观察层：CUPTI kernel active、CPU 串行 launch 外围的 CUDA-event effective timeline，以及同步 host wall。固定 6 个形状并用 3 个独立进程重复后，预分配输出的跨进程中位平均 effective timeline 约 `19.93us`（`19.70～23.65us`），每次分配输出约 `34.65us`（`34.35～40.61us`），而 CUPTI kernel active 中位数只有约 `1.464us`（`1.346～1.468us`）。active 仅占 effective timeline 的中位数约 `6.75%`，所以继续雕刻 gate 核内指令的全局上限很低；输出生命周期、launch 和与消费者的边界才是主要机会。

为验证这个判断，新增 matched synthetic consumer：物化路径先写出 FP32 `g/beta` 再由第二个 kernel 消费；融合路径在同一个 kernel 中计算并立即消费。三次独立进程均为 6/6 形状通过 `1e-5`，两核到一核的跨进程中位平均 effective timeline 为 `37.77us -> 20.44us`，平均加速中位数 `1.87x`（`1.84～1.91x`）；CUPTI active 总和为 `2.776us -> 1.591us`，平均加速中位数 `1.79x`（`1.77～1.79x`）。因此“直接 producer-consumer 融合”通过预设双重 `1.05x` 门槛，应晋级到 5090 目标机候选；该 synthetic consumer 不等价于 recurrent GDN，数字不得外推。

本机 Nsight Compute 2022.4.1 能连接目标进程，但硬件计数器被 `ERR_NVGPUCTRPERM` 权限阻断。原始日志和结构化 receipt 已保留，因此本轮不声称 occupancy、SFU、L2 或 DRAM 结论。详细实验设计、复现命令和读数方法见 `SM89_OPTIMIZATION_TUTORIAL.zh-CN.md`。

## 当前候选

首轮已经产出三个可打包实现，而不是继续空测：

1. `gate_fusion_auto`：用一个 Triton kernel 融合 `A_log/softplus/exp/sigmoid` 门控预处理，再调用 SM120 主体；目标是减少短序列上多个逐元素 launch。
2. `torch_force_cp`：保持精确门控表达式，强制 context-parallel，用于验证官方路由阈值是否保守。
3. `torch_force_non_cp`：保持精确门控表达式，强制普通路径，用于验证 CP 是否在部分形状上反而亏损。
4. `gate_fusion_runtime_extent_auto`：保持 gate 融合公式和原始 SM120 launch 几何，只抑制动态 `n_elements` 的值/对齐特化；4060 代理证据显示 specialization 从 5 降到 1，稳态差异约 1.13%，仍需在 5090 上裁决。

冻结 workload 显示，官方自动策略会将 82/100 例路由到 CP、18/100 例路由到非 CP。先验证这个高影响离散决策，比在未知瓶颈下连续微调 tile 更有信息价值。

这三个候选目前属于架构筛选实现，不等同于比赛可提交的完全自包含最终算子：其中主体仍调用 FlashInfer。若融合与路由实验确认收益，下一阶段应把胜出的调度假设移植到自包含 SM120 kernel，而不是把外部库调用包装成“自主 SOTA”。

## 能否达到理论最优或超过 SOTA

绝对理论最优通常不可证明，因为实现空间、编译器映射、动态频率和未观测硬件效应都没有封闭。可行的目标是给出三层边界：

1. 数学工作量和必要数据流给出不可突破的乐观下界；
2. 资源约束模型给出在 5090 带宽、计算、占用率和同步约束下的可达区间；
3. 与固定版本公开实现做同机、同输入、交错测量，给出到“已知最佳”的实测差距。

超过公开 SOTA 是可能事件，不是架构承诺。最有希望的路径不是拒绝外部知识，而是让外部知识提出不同架构、让 agent 自己生产实现、再由目标机证据否决或晋级。只有候选在 100/100 正确且同机全局分数稳定领先时，才可以说超过了比较对象；在此之前只能说“发现了有希望的局部或架构收益”。

## 尚未完成的严格验证

- GPU 独占窗口下的 100-case 正式基线；
- 三个候选在 6-case 架构筛选集上的正确性和交错性能；
- 幸存者的 100-case 全量正确性与全局等权分数；
- 胜出调度向自包含 SM120 kernel 的迁移；
- 与明确版本的公开最佳实现进行同机比较。

这些未完成项不会被“数学预估最优”替代。数学模型负责缩小搜索空间，生产精确测量负责裁决。
