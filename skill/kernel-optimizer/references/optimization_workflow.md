# Optimization workflow

The enforced sequence is `PLANNING -> BASELINE -> MODELING -> EXPERIMENT ->
PRODUCTION_VALIDATION -> CERTIFICATION -> COMPLETE`.  Use
`scripts/kernel_opt.py advance --check-only --to <phase>` to inspect the next gate and
rerun without `--check-only` only after every finding is resolved.  Do not edit
`run_state.json` by hand.

At every step run `scripts/kernel_opt.py next --run <run>`. Its action order is:

1. capture a correct production discovery baseline;
2. derive and rank 4--12 quantified global opportunities across at least four
   rewrite families;
3. bind 6--12 production candidates to at least three ranked opportunities and
   cover at least four architecture families;
4. repair compilation/correctness failures and cheaply screen anchor/edge cases;
5. promote at most 2--4 survivors;
6. close exact official hardware evidence and finalist-binary resource mapping;
7. calibrate P0 and identify the single uncertainty that can flip the finalist
   top-two ordering;
8. map that uncertainty to a defensible observable and choose screening or
   qualification;
9. materialize the bounded experiment and obtain independent supervisor approval;
10. execute and bind immutable evidence;
11. reconcile resource balance, schedule DAG, frontier and queue;
12. check the next phase gate.

The controller may safely create manifests and receipts. It must not execute an
incomplete command contract or interpret an arbitrary result without the global
scheduler's boundary-aware reconciliation.

The fast discovery lane is cross-cutting rather than a phase transition. After
a correct baseline exists, it may compile, repair, correctness-check and cheaply
screen run-local production candidates before full model closure. Every result
is labeled `DISCOVERY_ONLY`; only promoted finalists enter the enforced sequence
for supervised qualification and certification.

## Baseline gate

Freeze source hashes, ABI, inputs and environment.  Verify correctness before
timing.  Report CPU dispatch, individual GPU active time and end-to-end latency
separately.  Use warmup, clock/load checks, raw samples and an interleaved order
when comparing close candidates.

Before entering `MODELING`, every weighted workload case must have correctness,
CPU dispatch, GPU active and end-to-end baseline evidence with source and raw
sample identities.

## Hypothesis gate

Before constructing the qualification hypothesis, build a discovery portfolio
of 6--12 candidates across at least four architecture families. A compiler,
import, layout/type, harness or missing-artifact failure is technical and stays
repairable within its separate budget. It is not evidence against the
performance hypothesis. Use successive halving so every family receives a
small implementation budget before any finalist receives expensive modeling.

Change one scheduling/resource hypothesis at a time when possible.  State the
expected observable consequence in latency, instructions, transactions,
occupancy or stalls.  Select a particle benchmark that can falsify it.

Before entering `EXPERIMENT`, the run must contain a populated mandatory-work
ledger, mathematical/current DAG, target resource graph, initial SASS/resource
schedule and executable P0--P4 microbenchmark plan.  P0 measurement calibration
must already pass.

An experiment hypothesis is admissible only after candidate screening. The
decision contract must contain 2--4 candidates, the top two, their objective
intervals and a single uncertainty whose interval crosses the decision
boundary. The measurability contract must say how an observable estimates that
quantity, with confounders, controls, falsification and precision. If the
ordering cannot flip, stop measuring and implement or qualify the winner.

## Candidate gate

Require numerical correctness on primary and boundary workloads.  Audit source
identity, launch geometry, registers, spills, shared memory, static instruction
mix and applicable runtime counters.  Missing counters make the conclusion
weaker; they do not justify inventing attribution.

Complete `static/instruction_audit.json` before accepting a candidate.  Confirm
that final SASS contains the mechanism predicted by the hypothesis and map its
dependency/resource change back to the microarchitecture model.  PTX without
the launched binary is insufficient for this gate.

Before entering `PRODUCTION_VALIDATION`, P0--P3 evidence, cross-layer component
prediction, a correct accepted candidate and a matching final-binary audit must
pass.  Before `CERTIFICATION`, P4 must cover every weighted workload case and
the production-model residual must pass or be explicitly bounded.

## Decision gate

ACCEPT only when correctness passes and the weighted objective improves with a
stable paired confidence interval.  REJECT when correctness fails or evidence
contradicts the hypothesis.  Use INCONCLUSIVE for noise, environment drift or
missing discriminating evidence.  Preserve rejected candidates and reasons.

Stop when a defensible bound is reached, remaining hypotheses are below the
measurement resolution, or further work needs new authority/hardware.  Never
stop merely because one implementation beats a prior library.

Use cheap screening only to remove candidates. Qualification is reserved for
the surviving top two and uses production-matched controls. Both tiers have
pre-registered configuration, sample, process-launch, wall-clock and revision
budgets. A technical failure requires supervisor review; a causal rejection
requires a new frontier/decision contract. Sunk implementation effort is not a
reason to spend another sample.

Before closing a run, harvest eligible run-local microbenchmarks and execute the
repository-purity audit.  A failed promotion is not bypassed: either repair the
generic benchmark or explicitly classify the probe as application-shaped.
