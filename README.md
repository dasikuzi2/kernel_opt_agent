# Kernel Optimization Agent

For a human review of the framework boundary, execution flow, directory
ownership and contract map, start with [REVIEW.md](REVIEW.md). This README is
the operator quick start; `AGENTS.md` contains mandatory agent policy.
The design rationale and validation report for opportunity-driven search is
available in
[skill/kernel-optimizer/references/opportunity_driven_search_design.md](skill/kernel-optimizer/references/opportunity_driven_search_design.md).
The transfer-aware method-learning layer and its two-device validation are
documented in
[skill/kernel-optimizer/references/method_learning_design.md](skill/kernel-optimizer/references/method_learning_design.md).

This repository turns GPU-kernel optimization into a reproducible loop driven
by workload contracts, hardware evidence and falsifiable microbenchmarks.

It deliberately contains no application-specific algorithm, workload or
performance result.  Hardware facts are separated from empirical measurements;
measurements are keyed by device and software environment.

The workflow has two lanes. Fast discovery first compiles conditional model
terms into a ranked opportunity map, then writes and repairs a diverse set of
run-local production candidates linked to those opportunities. It uses cheap
anchor/edge screening and successive halving. Only survivors enter the evidence-closed
qualification and limit-certification lane. A technical build failure never
counts as a causal performance rejection.

## Start a run

An agent launched with this directory as its working tree is governed by
`AGENTS.md` and must request operator computation, workload and target hardware
before tuning.  The same gate is enforced by the command line:

```bash
python3 scripts/kernel_opt.py new-run --help
python3 scripts/kernel_opt.py new-run --print-intake
```

Once three manifests are available:

```bash
python3 scripts/kernel_opt.py new-run \
  --operator operator.json \
  --workload workload.json \
  --hardware hardware.json
```

`scripts/kernel_opt.py` is the stable public command surface. Individual
scripts remain implementation modules:

```bash
python3 scripts/kernel_opt.py --help
python3 scripts/kernel_opt.py new-run --operator operator.json --workload workload.json --hardware hardware.json
python3 scripts/kernel_opt.py next --run runs/<run-id>
```

After a correct discovery baseline is present, quantify several global
opportunities before managing the production-candidate portfolio:

```bash
python3 scripts/kernel_opt.py opportunity init --run runs/<run-id> --if-missing
python3 scripts/kernel_opt.py opportunity add --run runs/<run-id> --spec opportunity-spec.json
python3 scripts/kernel_opt.py opportunity rank --run runs/<run-id>
python3 scripts/kernel_opt.py opportunity close --run runs/<run-id> --opportunity-id <id> --disposition AT_MEASURED_ROOF --reason <reason> --evidence <result.json> --evidence-claim <claim> --reopen-condition <condition>
python3 scripts/kernel_opt.py opportunity reopen --run runs/<run-id> --opportunity-id <id> --reason <changed-condition>
python3 scripts/kernel_opt.py method recommend --run runs/<run-id>
python3 scripts/kernel_opt.py candidate init --run runs/<run-id> --if-missing
python3 scripts/kernel_opt.py candidate add --run runs/<run-id> --spec candidate-spec.json
python3 scripts/kernel_opt.py candidate run --run runs/<run-id> --candidate-id <id>
python3 scripts/kernel_opt.py candidate promote --run runs/<run-id> --candidate-id <id>
```

The default opportunity map requires 4--12 quantified opportunities across at
least four rewrite families. Each opportunity states the current global
contribution, a conditional optimistic gain ceiling, a likely gain interval,
confidence, implementation cost and hash-bound model evidence. Absolute-global-optimum labels are
rejected: a decomposition-specific minimum is not a semantic lower bound.
Candidates must bind to a ranked opportunity, stay below its gain ceiling and
cover at least three opportunities by default.
Measured dead ends can be marked `CLOSED` only with hash-bound run-local evidence,
a global stop reason and explicit reopen conditions. Closed opportunities score
zero and are excluded from method matching, candidate registration and next-action
routing; they return to the search budget only through an explicit audited reopen.

If that portfolio is still narrow, `method recommend` matches reusable method
cards against the frozen operator, workload, hardware and opportunity map. The
receipt is hash-bound to all four inputs and the card library. Literature and
vendor guidance remain discovery priors only: they cannot increase a gain
estimate, prove a hardware capability, accept a candidate or support a limit
claim. Unverified hard capabilities fail closed.

Discovery then requires 6--12 candidates across at least four architecture families
by default. The default discovery budget is two hours overall, twenty minutes
per candidate and eight technical repairs per candidate; expiry stops further
measurement for plan review. Candidates are ranked by weighted screening gain
and at most two are promoted by default. Screening records prediction-versus-
observation residuals. Its timing is a routing signal, not
production acceptance evidence.

Strict qualification is intentionally blocked until `hardware_evidence.json` archives exact
vendor-official documents for the programming model, ISA, target-architecture
tuning guide and device specification. If the agent cannot find one of those
official documents, the developer must provide its location; inferred hardware
facts and neighboring-device values are forbidden. Discovery-only production
implementation and cheap screening may proceed after a correct baseline; those
results cannot support a production acceptance or limit claim.

After the exact launched binary is archived inside the run, disassemble it with
a hash-bound tool/architecture receipt, classify every static instruction site,
and build the conservative resource set. Unknown or multiply classified SASS
mnemonics are a hard stop:

```bash
python3 scripts/kernel_opt.py sass-archive \
  --binary runs/<run-id>/static/launched.cubin \
  --output-sass runs/<run-id>/static/final.sass \
  --output-receipt runs/<run-id>/static/disassembly_receipt.json \
  --vendor NVIDIA --device-name '<exact device>' --compute-capability 12.0
python3 scripts/kernel_opt.py sass-count \
  --input runs/<run-id>/static/final.sass \
  --binary runs/<run-id>/static/launched.cubin \
  --disassembly-receipt runs/<run-id>/static/disassembly_receipt.json \
  --output runs/<run-id>/static/sass-summary.json
python3 scripts/kernel_opt.py resources-discover \
  --sass-summary runs/<run-id>/static/sass-summary.json \
  --hardware-evidence runs/<run-id>/hardware_evidence.json \
  --output runs/<run-id>/models/resource_discovery.json
python3 scripts/kernel_opt.py next --run runs/<run-id>
```

Use `scripts/kernel_opt.py hardware-discover` to create a hardware snapshot,
then use the selected microbenchmarks and analysis commands to build evidence. `runs/` is
for generated artifacts; reusable knowledge belongs in `hardware/`,
`knowledge/`, `microbench/`, `schemas/` or the skill references.

Each run designates one `GLOBAL_SCHEDULER` and an independent
`GLOBAL_SUPERVISOR`. The scheduler maintains the global resource balance,
2--4-candidate tradeoff frontier and candidate-driven experiment queue. The
supervisor alone approves a hash-bound, budgeted dispatch. Stage workers cannot
accept a local candidate or approve their own probe. Phase gates reject a
run whose material resources are omitted, whose unknown utilization is not
bound to an experiment request, or whose accepted candidate lacks a global
tradeoff decision.

The shortest legal experiment path is:

```bash
python3 scripts/kernel_opt.py experiment-rank --run runs/<run-id>
python3 scripts/kernel_opt.py experiment-materialize --run runs/<run-id> --request-id <id>
# Complete the sealed experiment.json, then independent review:
python3 scripts/kernel_opt.py experiment-approve --run runs/<run-id> --request-id <id> \
  --supervisor-id <registered-id> --rationale '<decision-boundary review>'
python3 scripts/kernel_opt.py experiment-dispatch --run runs/<run-id> --request-id <id>
```

Dispatch fails when the top-two ordering cannot flip, the quantity is not
identifiable at the required precision, any role identity overlaps, a tier
budget is exceeded, or any approved artifact changed.

Run phases are non-skippable.  Inspect and advance the next gate with:

```bash
python3 scripts/kernel_opt.py advance --run runs/<run-id> --to BASELINE --check-only
python3 scripts/kernel_opt.py advance --run runs/<run-id> --to BASELINE
```

The enforced order is planning, production-exact baseline, modeling,
P0--P3 experiments, production/P4 validation, certification and completion.

## Clean asset lifecycle

Each directory has one owner:

- `runs/` contains mutable application work, raw evidence, binaries and new
  microbenchmark candidates.
- `microbench/` contains promoted application-independent source packages only.
- `hardware/measurements/` contains immutable results keyed by complete device
  and software identity.
- `skill/`, `scripts/`, `schemas/` and `templates/` contain only reusable
  instructions, automation and contracts.

Create candidates with `scripts/kernel_opt.py microbench-new`. At each accepted
hypothesis and before closing a run, execute `scripts/kernel_opt.py
microbench-harvest --run <run> --promote`; only candidates whose
correctness, controls, clean build, two independent cold starts, genericity and
static-instruction evidence are hash-bound and pass are added to the catalog.
Mechanism, device-calibrated and production-predictive claims have progressively
stronger gates. Device qualification additionally requires an
`EVIDENCE_CLOSED_V2` record in `hardware/measurements/index.json`; a string
`PASS` or an unregistered result is rejected. Promotion is append-only and never overwrites an
existing package. `scripts/kernel_opt.py audit` rejects undeclared files,
caches, generated outputs, production dependencies and task-specific content
in reusable directories.

Cold validation commands are argv-form JSON executed by `scripts/kernel_opt.py
microbench-reproduce`. The executor rejects source-tree
outputs and stale pre-existing artifacts, then binds logs and fresh outputs to
its own identity. Promotion accepts PASS check results only when those exact
artifacts occur in a trusted reproduction receipt.

## Evidence classes

- `FACT`: queried or statically verified.
- `MEASURED`: backed by immutable raw samples.
- `INFERRED`: derived from stated facts and measurements.
- `HYPOTHESIS`: awaiting a discriminating experiment.
- `REJECTED`: falsified or measured with an invalid method.

See `skill/kernel-optimizer/references/` for the optimization protocol.

## Seeded hardware evidence

The first adapter and historical dataset target an RTX 5090 / SM120 environment.
They live under `hardware/measurements/nvidia/rtx5090_sm120_gpu6/` and include a
hardware snapshot, raw launch/barrier/load/store samples, service-curve fits,
the compiled binary identity, resource usage and SASS.  The counter-access
probe is archived separately and currently reports `DENIED`; no stall-counter
claim is permitted from that environment.

Those historical records are explicitly `LEGACY_UNQUALIFIED`: they may be
inspected, but they cannot parameterize a hardware model. A usable measurement
must be re-run and registered as `EVIDENCE_CLOSED_V2` with official target
evidence, P0 calibration, source, binary, final SASS and raw samples. New device
models receive separate snapshots and measurement directories rather than
inheriting values.
