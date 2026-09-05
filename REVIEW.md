# 框架人工审核导航

这是审核本仓库的最短路径。它只描述通用算子优化框架，不描述某一个
具体算子，也不把历史优化结论当作框架事实。

## 1. 系统边界

框架接收三项冻结输入，最终给出带证据等级的优化结论：

```text
算子计算契约 + 加权目标工作负载 + 精确目标硬件
                         │
                         ▼
规划 → 正确基线 → 全局机会排序 → 多架构候选发现 → 建模/严格实验
                         │
                         ▼
生产验证 → 极限证书 → 中文人工审核报告
```

一次更快的计时结果不等于达到理论极限。极限证明必须同时具备：目标机
不可变证据、必要工作量、资源服务曲线下界、依赖 DAG 下界、可行调度上界、
实测置信区间、最终二进制 SASS，以及按每个工作负载重新计算的差距。

## 2. 建议审核顺序

| 顺序 | 文件 | 审核问题 |
|---:|---|---|
| 1 | `AGENTS.md` | 哪些约束不可跳过，哪些行为被禁止？ |
| 2 | `skill/kernel-optimizer/SKILL.md` | Agent 如何根据阶段加载规则并推进工作？ |
| 3 | `scripts/kernel_opt.py` | 对外命令是否完整、唯一且职责清楚？ |
| 4 | `scripts/new_run.py` | 算子、负载和硬件三项输入如何冻结？ |
| 5 | `scripts/opportunity_map.py`、`scripts/candidate_discovery.py`、`scripts/advance_run.py` | 全局机会如何量化排序；候选如何快速产生、修复和筛选；每个严格阶段依靠哪些证据才能关闭？ |
| 6 | `schemas/*.schema.json` | 哪些结果被持久化并由机器严格验证？ |
| 7 | `tests/test_evidence_closed_workflow.py` | 一条完整正向证据链能否跑通？ |
| 8 | `tests/test_repository.py` | 不完整、越权或伪造状态是否会失败关闭？ |

其余内容都是为上述八个审核点服务的实现模块、模板、可复用探针或不可变
测量包。

## 3. 唯一公开执行流

正常使用只调用：

```bash
python3 scripts/kernel_opt.py <command> ...
```

`scripts/` 下的其他 Python 文件都是实现模块，不是第二套公开接口。

| 阶段 | 公开命令 | 主要落盘结果 |
|---|---|---|
| 输入冻结 | `new-run`、`hardware-*` | 三项输入契约和官方目标硬件证据 |
| 规划 | `sass-archive`、`sass-count`、`resources-discover` | 最终二进制身份和完整候选资源集合 |
| 基线 | `p0-calibrate`、`paired-compare` | 生产一致的 CPU、GPU 和端到端基线 |
| 机会与候选发现 | `opportunity init/add/rank/close/reopen`、`method validate/recommend`、`candidate init/add/run/promote` | 条件收益上界及价值排序；哈希绑定死路关闭与显式重开；可迁移方法先验；6--12 个跨架构族、跨机会生产候选、技术修复和预测残差 |
| 建模 | `service-curve-fit`、`next`、`advance` | 必要工作、DAG、调度、资源平衡和实验队列 |
| 实验 | `experiment-*` | 密封执行、绑定结果、候选决策和模型闭环 |
| 经验复用 | `microbench-*` | 通过资格审查的应用无关原子探针 |
| 证明与展示 | `certify`、`report-validate`、`report-render` | 重算的极限证书和中文审核 HTML |

阶段状态机不可跳过：

```text
PLANNING → BASELINE → MODELING → EXPERIMENT
         → PRODUCTION_VALIDATION → CERTIFICATION → COMPLETE
```

## 4. 目录的唯一职责

| 目录 | 唯一职责 | 禁止混入 |
|---|---|---|
| `runs/` | 当前算子的可变工作、原始证据和候选实现 | 可复用硬件事实 |
| `hardware/adapters/` | 厂商证据接纳策略 | 性能测量结果 |
| `hardware/specs/` | 有官方证据的硬件契约 | 未记录来源的推断值 |
| `hardware/measurements/` | 按完整身份保存的不可变测量包 | 可变运行状态 |
| `microbench/` | 已提升的、应用无关的原子探针 | 生产依赖、二进制和原始样本 |
| `schemas/` | 可持久化产物的机器契约 | 示例和运行输出 |
| `scripts/` | 确定性实现模块和唯一公开 CLI | 具体算子源码和结果 |
| `templates/` | 新运行和报告的初始结构 | 已接受证据或伪装成事实的默认值 |
| `skill/` | Agent 路由规则和按需读取的参考资料 | 生成报告和测量结果 |
| `tests/` | 失败关闭的契约与工作流测试 | 生产数据 |

`scripts/kernel_opt.py audit` 负责检查这些可复用区域的纯净性。

## 5. `scripts/` 的职责分组

`scripts/` 当前共有 47 个文件。为了保持直接执行和同级导入的确定性，物理
目录保持扁平；逻辑所有权严格分组如下：

- 运行生命周期：`new_run.py`、`optimizer_step.py`、`advance_run.py`、
  `audit_repository.py`。
- 机会编译：`opportunity_map.py`。把模型项转成带条件作用域、收益区间、
  置信度和实现成本的排序对象，并拒绝伪装成绝对最优的分解下界。
- 候选发现：`candidate_discovery.py`。候选必须绑定已排序机会；技术失败进入可修复循环，只有通过
  anchor/edge 正确性和廉价性能筛选的候选才会送入严格资格验证。
- 硬件证据：`discover_hardware.py`、`init_hardware_evidence.py`、
  `add_official_hardware_source.py`、`add_documented_hardware_fact.py`、
  `validate_hardware_evidence.py`、`register_measurement.py`、
  `probe_ncu_access.py`、`cuda_device_query.cu`。
- 最终二进制：`archive_final_binary_sass.py`、`count_sass.py`、
  `discover_resources.py`。
- 测量与拟合：`calibrate_p0.py`、`fit_service_curve.py`、
  `compare_paired.py`。
- 实验事务：`rank_experiments.py`、`materialize_experiment.py`、
  `dispatch_experiment.py`、`execute_experiment.py`、
  `bind_experiment_result.py`、`apply_model_updates.py`、
  `reconcile_experiment_result.py`。
- Microbenchmark 生命周期：`query_microbench_catalog.py`、
  `new_microbench_candidate.py`、`execute_microbench_reproduction.py`、
  `promote_microbench.py`、`harvest_microbenches.py`。
- 证明与展示：`emit_certificate.py`、`validate_human_review_report.py`、
  `render_human_review_report.py`。
- 仅供共享校验：`schema_utils.py`、`evidence_utils.py`、
  `experiment_utils.py`、`model_patch_utils.py`、`p0_utils.py`、
  `repository_rules.py`。

所有公开命令及其一句话职责可直接查看：

```bash
python3 scripts/kernel_opt.py --help
```

## 6. Schema 职责分组

- 运行与输入：`run_state`、`operator_contract`、`workload`、
  `hardware_snapshot`。
- 机会与候选发现：`opportunity_map`、`optimization_method`、
  `method_match_receipt`、`candidate_pool`、`candidate_smoke_result`。
- 官方证据与测量：`hardware_evidence_manifest`、
  `p0_calibration_receipt`、`benchmark_result`。
- 规划与建模：`optimization_plan`、`microarchitecture_model`、
  `mandatory_work_ledger`、`operator_dag`、`resource_discovery`、
  `resource_schedule_model`、`resource_balance_ledger`、
  `tradeoff_frontier`、`global_schedule_state`、`microbenchmark_plan`、
  `instruction_audit`、`cross_layer_model_validation`。
- 密封实验事务：`experiment_request`、`executable_experiment`、
  `experiment_execution_receipt`、`candidate_decision`、
  `model_update_plan`、`semantic_model_update_receipt`。
- 可复用探针：`microbenchmark_definition`、
  `microbenchmark_promotion`。
- 生产验证与证明：`production_baseline`、`production_validation`、
  `achieved_performance`、`limit_bound`、`architecture_explanation`、
  `limit_certificate`、`human_review_report`。

以上名称均对应 `schemas/<name>.schema.json`。

## 7. 最值得质疑的信任边界

1. 硬件事实只能来自精确官方资料或目标设备官方查询，必须带定位信息和
   SHA-256 身份。
2. 物质资源集合必须由最终二进制指令类别和官方映射产生，手写资源列表不能
   关闭规划阶段。
3. 实验只有在源码、argv 命令、控制变量、预期 SASS 和产物路径全部密封后
   才能执行。
4. 模型更新必须是字段级变换，更新前后值都能从已绑定结果重新计算。
5. 单阶段更快不等于全局接受；全局调度者必须同时闭合资源平衡、调度模型和
   算力—访存权衡前沿。
6. 可复用 microbenchmark 必须通过独立冷启动复现和纯净性检查，运行内探针
   不能直接复制进公共目录。
7. 最终证书重新计算每个工作负载的上下界和加权差距，不信任预先填写的结论。

## 8. 与框架分离的历史数据

当前仓库的 `runs/` 为空。旧契约运行记录和退役文件保存在仓库外：

```text
/workspace/dance/qwen35/kernel_opt_agent_run_archive/20260826_pre_evidence_closed_v2
```

`hardware/measurements/` 中预置的 RTX 5090 数据明确标记为
`LEGACY_UNQUALIFIED`：它可以用于历史检查，但在按当前证据契约重跑并登记前，
不能参数化新的硬件模型。

## 9. 当前审核热点

- `scripts/advance_run.py` 是集中式阶段门禁引擎，体积较大，但每个阶段的边界
  已显式分区，并由失败关闭测试覆盖。
- `tests/test_repository.py` 是广覆盖负向契约测试；
  `tests/test_evidence_closed_workflow.py` 是较短的正向完整证据链。
- 框架运行时不依赖外部 Python 包；目标工具只允许由密封命令调用。JSON
  Schema 使用 `scripts/schema_utils.py` 中明确支持的子集进行验证。

如果审核重点是可维护性、门禁完整性或伪造证据能否被误接纳，优先检查以上
三处。
