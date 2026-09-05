# Kernel optimization repository instructions

These instructions apply to every agent working inside this repository.

## Mandatory intake gate

At the start of every new optimization task, show the user a concise reminder
that three inputs are required and ask for any missing fields:

1. Operator computation: equations or pseudocode, inputs/outputs/state,
   shapes/strides/dtypes, numerical contract, aliasing and legal rewrites.
2. Target workload: shape set, occurrence weights, execution modes, upstream
   and downstream layouts, concurrency, graph mode and latency objective.
3. Target hardware: exact device or permission to auto-discover it, software
   stack, power/clock policy, allowed programming models and architecture-
   specific features.

Do this even when historical defaults exist.  Historical input may be reused
only after the user names a run or explicitly confirms reuse.  Read-only
inspection may continue while fields are missing, but do not compile, tune,
benchmark or change production code before the intake gate is complete.

Once the three inputs are complete and the requested optimization scope is
authorized, create the run and begin baseline/model construction immediately.
Do not pause merely to ask the user what to do next.  Ask again only when a
mathematical choice, correctness relaxation, expensive experiment or external
mutation requires new authority.

## Optimization invariants

- Freeze a machine-readable operator contract, workload and hardware snapshot
  before establishing the baseline.
- Before recording any target-hardware capability, resource mapping or numeric
  specification, archive an exact vendor-official source or official target-
  device query with URL/command, version, locator and SHA-256. Search official
  sources first. If the exact architecture/device is not documented clearly,
  ask the developer for the official document location and stop hardware-model
  construction. Never infer from a neighboring architecture or product.
- After a correct production baseline exists, create and rank 4--12 quantified
  global opportunities across at least four rewrite families. Distinguish
  decomposition-conditional work, current-schedule work and empirical
  bottlenecks; never label one of them an absolute global optimum. Rank by
  expected global gain, confidence and implementation cost before measuring.
- Bind every discovery candidate to a ranked opportunity and a predicted global
  gain interval, then write and cheaply screen 6--12 run-local production
  candidates across at least four materially different architecture families
  and at least three opportunities. Full resource-model closure is
  not required for discovery-only compilation, correctness and anchor/edge
  timing. Read `skill/kernel-optimizer/references/discovery_loop.md`.
- Promote at most 2--4 discovery survivors into supervised qualification. For
  those finalists, build the explicit optimization plan and target-
  microarchitecture resource graph, compute candidate-specific objective
  intervals and identify exactly one unknown whose interval can flip the top-
  two ordering. `UNKNOWN` in a resource ledger does not itself authorize a
  qualification measurement.
- Separate `GLOBAL_SCHEDULER`, `MICROARCHITECTURE_ANALYST`, `EXPERIMENT_AGENT`
  and `GLOBAL_SUPERVISOR` actor identities. The supervisor alone approves
  dispatch and owns veto, budget, stop and replan authority.
- Use an atomic microbenchmark only when a measurability contract shows that
  its observable identifies the decision quantity with sufficient precision.
  Otherwise use candidate A/B, existing evidence or no measurement.
- Do not authorize hardware measurement merely because a model field is
  unknown. Until opportunity-linked production code survives smoke screening,
  the next action is implementation or repair. Record the prediction residual
  after screening and use it to recalibrate subsequent opportunity estimates.
- Separate screening from qualification. Freeze configuration, sample,
  process-launch, wall-clock and revision budgets before materialization.
- Separate technical repair from causal revision. Compiler/import/layout/type,
  harness and missing-artifact failures consume a bounded technical-attempt
  budget and remain repairable. They never reject a performance hypothesis and
  never consume the decision contract's causal-revision budget.
- Preserve the mathematical result and public ABI unless the user authorizes a
  change.  Record every authorized relaxation explicitly.
- Separate mathematical DAG edges from schedule-induced serialization.
- Maintain a mandatory-work ledger before proposing a lower bound.
- Time native GPU kernels with a method that excludes CPU enqueue gaps.  Treat
  end-to-end CPU time and GPU active time as separate metrics.
- A benchmark is comparable only when source identity, ABI, workload, launch
  geometry, clocks, competing load and measurement semantics are recorded.
- Every qualification candidate ends in ACCEPT, REJECT or INCONCLUSIVE with raw
  evidence. Discovery candidates end in a discovery lifecycle state such as
  SCREENED_OUT, TECHNICALLY_BLOCKED or PROMOTED_TO_QUALIFICATION.
- Every accepted candidate with inspectable device code requires a final-binary
  PTX/SASS/resource audit.  Source syntax or PTX alone does not prove which
  machine instructions ran.
- Map mandatory work and instruction dependency chains to warp schedulers,
  issue paths, register/shared-memory resources, compute pipelines, memory
  paths, synchronization and architecture-specific engines that are relevant
  to the target.  Unknown properties stay explicit.
- Never call a theoretical peak, a calibrated service rate and an achieved
  production time the same kind of limit.
- Never turn an operator-specific measurement into a hardware fact.  Hardware
  measurements must come from standalone synthetic microbenchmarks.
- Generate the material-resource candidate set from final-binary instruction
  classes plus the official-source manifest with
  `scripts/kernel_opt.py resources-discover`.
  A hand-written resource list, unresolved mapping or missing official document
  is a planning-gate failure.
- Treat a missing or unknown framework contract as a hard failure. Production
  runs use `optimization-run-state-v4` plus `evidence-closed-v2`; only explicit
  synthetic TEST fixtures may exercise legacy validation.
- Execute only a sealed argv-form experiment contract. Raw samples, result,
  static audit and reproduction log must be created or changed by that
  execution, then hash-bound before result binding.
- Apply experimental model changes through field-level transforms whose input,
  before value, after value, units and uncertainty can be recomputed from the
  bound result. Resource balance, schedule and tradeoff frontier must all be
  reconciled before reranking.
- A proof claim requires per-case immutable silicon, resource-service and DAG
  lower bounds, a feasible-schedule upper bound, achieved confidence intervals
  and a recomputed workload-weighted gap. SASS explanation without those bounds
  may explain a residual but cannot prove a theoretical limit.

## Global scheduling and supervision authority

Every run has exactly one global scheduling/resource-modeling owner and one
independent global supervisor. In a
multi-agent workflow this is a dedicated global scheduler; in a single-agent
workflow the active agent must explicitly hold the same role.  The role is an
artifact and decision boundary, not an optional staffing convention.

Only the global scheduler may construct and rank the candidate frontier, declare a resource
model closed, accept a schedule candidate or authorize a human limit report.
Stage/kernel agents may propose hypotheses, implement candidates and return raw
evidence, but they must not optimize a local stage by silently worsening a
different resource or stage.

Only the global supervisor may approve or veto dispatch. The exact experiment,
decision contract, measurability contract, objective, frontier and tier budget
are hash-bound in `supervisor_approval.json`; an edit invalidates approval. A
technical failure enters `AWAITING_SUPERVISOR_REVIEW`; a causal rejection enters
`HALT_AND_REPLAN`. Neither state may automatically return to `PLANNED`.

The global scheduler owns `models/global_schedule_state.json`,
`models/resource_balance.json`, `models/tradeoff_frontier.json` and
`models/experiment_queue.json`.  Read
`skill/kernel-optimizer/references/global_scheduler.md` before plan
construction or delegation.  Missing ownership, resource coverage, utilization
semantics, tradeoff accounting or model-driven experiment requests is a phase-
gate failure, not a documentation omission.

Use `scripts/kernel_opt.py opportunity` for the quantified global opportunity
map, then `scripts/kernel_opt.py candidate` for the fast discovery portfolio and
repair loop. Discovery evidence can only route a candidate into qualification;
it cannot accept production performance or support a limit claim.
When an opportunity reaches a measured roof or a global materiality stop, close
it with `opportunity close`; free-form notes do not stop scheduling. A closure
must hash-bind run-local evidence and list explicit reopen conditions. Never
register or resume a candidate against `CLOSED` until `opportunity reopen`
records which condition changed.

Use `scripts/kernel_opt.py experiment-rank` for a reproducible finalist-
specific decision-value ranking receipt. Use `experiment-materialize`,
`experiment-approve` and `experiment-dispatch`;
`DISPATCHED` is forbidden until source,
commands, parameter matrix, controls, expected final SASS and artifact paths
are hash-bound and executable. Bind results with `experiment-bind`, then use
`experiment-apply` and `experiment-reconcile` before executing another
hypothesis. All command names in this paragraph use the public
`scripts/kernel_opt.py` entrypoint.

## Required run artifacts

Use the public `scripts/kernel_opt.py` command surface for normal operation;
direct script entrypoints are implementation modules. Use `new-run` to create
a run. Keep raw samples immutable and derive
summaries from them.  A completed run contains the frozen inputs, baseline,
optimization plan, opportunity map, microarchitecture model, work ledger,
  mathematical/current DAGs, global scheduling state, per-resource balance,
  compute-memory tradeoff frontier, model-driven experiment queue,
  per-candidate instruction audits, model-driven experiment requests and candidate decisions,
  environment identity, reproduction command and a limit certificate.

Every run advances only through `scripts/kernel_opt.py advance` in this order:

`PLANNING -> BASELINE -> MODELING -> EXPERIMENT -> PRODUCTION_VALIDATION ->
CERTIFICATION -> COMPLETE`.

Do not hand-edit phase state or perform qualification/certification work
belonging to a later phase. Discovery is a cross-cutting, explicitly
`DISCOVERY_ONLY` lane after a correct baseline and may run before full modeling
closure. The
run maintains production baselines, a P0--P4 microbenchmark plan, a calibrated
SASS/resource schedule, cross-layer prediction validation and production
validation.  `NOT_APPLICABLE` requires evidence and cannot bypass P0, P1, P3 or
P4 for a performance-limit claim.

## Reusable asset boundary

- Develop new probes only under a run's `microbench_candidates/` directory.
- Automatically attempt promotion when a probe becomes
  application-independent.  Use the promotion and repository-audit scripts;
  failed checks leave the probe run-local.
- Promotion uses structured, hash-bound check results and at least two
  independent cold-start receipts. Device-calibrated status additionally
  requires a registered `EVIDENCE_CLOSED_V2` hardware measurement; historical
  or self-declared `PASS` records cannot parameterize a model.
- `microbench/` contains only promoted definitions and source.  It must never
  contain raw samples, profiles, binaries, caches, production imports or
  application-specific names and paths.
- Published benchmark packages and registered measurement bundles are
  append-only.  Create a new version instead of overwriting one.
- Run `scripts/kernel_opt.py audit` after promotion and before completing a
  run.  Directory purity is a release gate.

Read `skill/kernel-optimizer/SKILL.md` for routing.  Load only the reference
needed for the current phase.
