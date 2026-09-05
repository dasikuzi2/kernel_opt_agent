---
name: kernel-optimizer
description: Plan, model and optimize GPU kernels against an explicit computation contract, target workload and target hardware using microarchitecture-rooted resource models, PTX/SASS audits, reproducible microbenchmarks and evidence-graded limit certificates. Use for kernel performance analysis or iterative implementation; do not use for ordinary application profiling without kernel optimization intent.
metadata:
  short-description: Evidence-driven GPU kernel optimization
---

# Kernel optimizer

Begin every new run with the mandatory intake gate.  Ask the user to provide or
confirm operator computation, target workload and target hardware.  Do not tune
until all three are frozen.  Read [references/intake.md](references/intake.md)
when fields are missing or a historical run may be reused.

Then execute an evidence-driven loop:

1. Freeze semantics, ABI, correctness tolerances and workload weights.
2. Discover and snapshot hardware/toolchain state, then establish a correct
   production-exact discovery baseline.
3. Read
   [references/discovery_loop.md](references/discovery_loop.md), compile 4--12
   conditional global opportunities from the work/DAG models and rank them by
   expected global gain, confidence and implementation cost with
   `scripts/kernel_opt.py opportunity`. Never present a decomposition-specific
   floor as an absolute global optimum.
4. Generate 6--12 materially different architecture candidates linked to at
   least three ranked opportunities and write their run-local production
   implementations. Use `scripts/kernel_opt.py candidate` to give
   compiler, import, layout and harness failures a bounded repair loop.
5. Cheaply screen every valid architecture family on an anchor and edge case.
   Discovery results route work only; they do not accept a candidate or claim a
   hardware fact. Promote at most 2--4 survivors.
6. For qualification finalists, build `hardware_evidence.json` from exact
   vendor-official documents and official target-device queries. Archive URL,
   command, version, section, artifact and SHA-256. Do not record an inferred
   hardware fact or a neighboring-device value.
7. Archive the exact launched finalist/baseline binaries with
   `scripts/kernel_opt.py sass-archive`, classify them with `sass-count`, then
   run `resources-discover`. Every static site must map to exactly one reviewed
   instruction class; unresolved or ambiguous mappings block qualification.
8. Create the optimization plan and target-
   microarchitecture resource graph. Build mandatory-work and
   mathematical/current-DAG ledgers, then map the current schedule onto the
   resource graph.
9. Bind and freeze the 2--4 promoted architecture-level candidates. Compute each
   candidate's resource-constrained objective interval.  Register only the one
   unresolved quantity whose uncertainty can change the top-two ordering; an
   `UNKNOWN` resource is not by itself permission to measure.
10. Have a separate microarchitecture analyst map that abstract quantity to an
   observable.  If an atomic probe cannot identify it with the precision
   required by the decision boundary, use candidate A/B or stop; do not expand
   a parameter sweep.  Read
   [references/decision_supervision.md](references/decision_supervision.md).
11. Pass P0 from raw positive/zero-work, graph/direct, clock, competing-load,
   independent-process and cold/warm measurements. Then materialize the
   experiment contract before dispatch: immutable source, argv-form commands,
   parameter matrix, controls, expected SASS and artifacts.
12. Obtain a hash-bound `GLOBAL_SUPERVISOR` approval.  The supervisor must be
   distinct from the scheduler, microarchitecture analyst and experimenter,
   may veto the experiment, and enforces separate screening/qualification
   budgets.  No approval means no dispatch.
13. Qualify one discovery survivor, validate full-workload correctness, run
   interleaved paired measurements and complete the required PTX/SASS/resource
   audit. Technical implementation failures are repaired in discovery and do
   not count as causal experiment revisions.
14. Record ACCEPT, REJECT or INCONCLUSIVE; bind only results created by the
   sealed execution contract. Apply field-level update transforms to resource
   balance, schedule and frontier, verify before/after hashes and recompute the
   changed fields from the bound result before reranking.
15. When a new probe answers an application-independent hardware question,
   automatically process it through the microbenchmark promotion gate. Static,
   mechanism, device and production-predictive qualification are distinct;
   device claims require an evidence-closed measurement registration. Keep a
   failed or application-shaped probe inside its run; never copy it directly
   into the reusable catalog.

Every run must designate exactly one global scheduling/resource-modeling owner
and one independent global supervisor.  Before building the plan or delegating stage work, read
[references/global_scheduler.md](references/global_scheduler.md).  The global
scheduler owns resource balance, candidate generation, compute-memory
tradeoffs and ranking.  The supervisor alone approves dispatch and may halt or
force replanning.  Stage agents return evidence; they do not independently
declare a local optimum.

Use `scripts/kernel_opt.py advance` for every phase transition. A production run must
use the `evidence-closed-v2` contract; an absent or unknown contract never falls
back to legacy behavior. Do not qualify a discovery candidate before the run
reaches `EXPERIMENT`, validate production
before `PRODUCTION_VALIDATION`, or issue a limit claim before `CERTIFICATION`.
Run-local discovery implementation and cheap screening are allowed after a
correct baseline, regardless of formal phase, because they carry no production
acceptance claim.
The phase checker is a minimum gate; passing it does not replace technical
judgment or evidence review.

Prefer the stable `scripts/kernel_opt.py` command surface; individual scripts
are implementation modules. Use `scripts/kernel_opt.py next --run <run>` after
every artifact change. It
selects the next model-driven action; `--apply-safe` may perform deterministic
ranking and planning steps but never executes an unmaterialized experiment or
silently applies an arbitrary numeric result to the global model.

For plan construction and resource mapping, read
[references/microarchitecture_planning.md](references/microarchitecture_planning.md),
then use [references/modeling.md](references/modeling.md) for lower bounds.  For
PTX/SASS admission and causal instruction analysis, read
[references/instruction_analysis.md](references/instruction_analysis.md).  For
test design and decision gates, read
[references/optimization_workflow.md](references/optimization_workflow.md).
When creating or reusing a microbenchmark, read
[references/microbenchmark_lifecycle.md](references/microbenchmark_lifecycle.md).
For timer choice, mechanism controls, P0--P4 qualification and coupled-resource
experiments, also read
[references/microbenchmark_precision.md](references/microbenchmark_precision.md).
For evidence labels and invalid measurement patterns, read
[references/evidence_grades.md](references/evidence_grades.md).  For hardware
selection and database rules, read
[references/hardware_routing.md](references/hardware_routing.md).

When producing a human-facing optimization review, read the complete contract
under [references/human_review_contract/](references/human_review_contract/README.md).
Build the machine-readable report first, validate it with `scripts/kernel_opt.py
report-validate`, and only then render HTML with `report-render`. The primary page must use the
contract's Chinese vocabulary, keep internal experiment IDs in the collapsed
developer appendix, distinguish capacity from activity, and map every latency
or throughput claim to a matched measurement and an explicit boundary.

Use repository scripts for deterministic intake, discovery, fitting, paired
analysis, benchmark promotion, repository-purity auditing and certificate
emission. A proof certificate is computed per workload case from immutable
silicon, resource-service and dependency-DAG lower bounds plus a feasible
schedule upper bound; the checker recomputes the weighted gap. `runs/` owns
mutable and application-specific work. Reusable
directories contain only promoted source, schemas, instructions or immutable
hardware evidence; do not place build products or production imports there.

After intake is complete, start the authorized baseline and modeling work
without asking for another generic confirmation.  Pause only for a missing
mathematical choice, correctness relaxation, costly experiment or external
mutation that exceeds the user's authority.
