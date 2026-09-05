# Fast production-candidate discovery

The discovery lane exists to produce working production candidates quickly. It
does not produce an acceptance claim, a hardware fact or a limit certificate.
Those remain owned by supervised qualification and certification.

## Portfolio before polishing

After a correct production baseline exists, first create 4--12 opportunity
records with `kernel_opt.py opportunity add`, then rank them. Each record must
name the source model term, whether it is decomposition-conditional,
current-schedule or empirical, its current objective contribution, optimistic
gain ceiling, likely gain interval, confidence, rewrite families and an
implementation budget. Source model artifacts are run-relative and SHA-256-
bound; a changed ledger invalidates the ranking. The numeric invariant is `0 <= likely lower <= likely
upper <= optimistic ceiling <= current contribution`.

An opportunity map is a search prior, not a proof of a global optimum. In
particular, work required by the current four-stage decomposition may disappear
under legal fusion. The `ABSOLUTE_GLOBAL_OPTIMUM` scope is therefore forbidden
for opportunity records.

After ranking, generate 6--12 candidates across
at least four materially different architecture families. Vary mathematical
decomposition, fusion boundaries, materialization, CTA/warp ownership,
register/shared-memory dataflow, persistent scheduling, instruction mechanism
or workload specialization. Parameter variants of the same schedule count as
one family.

Each candidate must bind to one ranked opportunity, use one of its rewrite
families and state a predicted global-gain interval below that opportunity's
ceiling. Cover at least three opportunities by default rather than producing
many variants of the same hypothesis. Give every family a small implementation budget before
spending qualification effort on any one family.

Before registration, every candidate must include a hash-bound
`dependency_contract` with status `PROVEN_LEGAL`. It records the mathematical
dependencies that remain, the implementation boundaries that change, forbidden
rewrites, numerical-ordering constraints and the source evidence used for that
decision. A literature or profiler match is not a legality proof. If the
contract cannot explain why the rewrite preserves the operator DAG, do not
compile it; re-audit the dataflow or choose another opportunity.

When the portfolio lacks architecture diversity, run `kernel_opt.py method
recommend --run <run>`. It matches reusable, source-attributed method cards to
the frozen operator, workload, hardware and ranked opportunity map. A match is
only a `DISCOVERY_PRIOR_ONLY` candidate-generation hint: it cannot increase a
modeled gain, validate a hardware capability, accept a candidate or support a
limit claim. Missing hard capabilities fail closed, architecture affinities
outside the source scope require adaptation, and every recommendation receipt
is hash-bound to both run inputs and the reusable card set.

Turn a matched method into one or more run-local production candidates, not a
literature summary. Preserve its stated failure modes, bottleneck shifts and
validation recipe in the candidate hypothesis. If no method applies, widen the
opportunity/decomposition analysis rather than forcing a fashionable technique
onto the operator.

## Repairable implementation loop

Candidate source lives under `runs/<run>/candidates/<id>/`. Register an argv-form
build, correctness and smoke command with `kernel_opt.py candidate add`, then use
`kernel_opt.py candidate run`.

A compiler error, import error, layout/type mismatch, missing build artifact or
invalid smoke harness is a `TECHNICAL_FAILURE`. It may be repaired repeatedly
within the candidate's technical-attempt budget. It does not consume the
decision contract's causal-revision budget and must never be recorded as a
performance rejection.

The smoke test uses one representative anchor and one edge case, minimal warmup
and a small sample count. Its result is discovery-only. A survivor is promoted
to the existing sealed A/B qualification flow, which reruns production-matched
correctness, timing and final-binary audits.
The observed global gain and prediction residual are written back to the
opportunity map so later estimates can be recalibrated.

## Close measured dead ends explicitly

An observation that says “reject” does not remove an opportunity from the
scheduler. Use `kernel_opt.py opportunity close` only when run-local evidence
supports a global stop disposition such as a measured rejection, a measured
service roof, a materiality floor, or a hard dependency block. The closure
certificate must include the evidence SHA-256 and concrete reopen conditions.

Closed opportunities score zero and are omitted from method matching,
candidate registration and next-action routing. If every opportunity is
closed, the only valid discovery action is `OPPORTUNITY_PORTFOLIO_CLOSED`.
Resume with `kernel_opt.py opportunity reopen` only after naming the changed
condition; the event remains in the map so the agent cannot silently repeat a
previous dead end.

## Successive halving

Use discovery budget in this order:

1. compile and minimal correctness for every architecture family;
2. cheap anchor/edge screening for every valid implementation;
3. retain at most four candidates for broader screening;
4. promote at most two candidates to supervised qualification;
5. apply full resource modeling and limit certification only to finalists.

Promotion is blocked while any registered candidate remains proposed or under
repair, so a convenient early result cannot suppress unexplored families.

Do not build a new atomic microbenchmark when a direct candidate smoke test can
eliminate a candidate more cheaply. Do not wait for every resource-model field
to close before writing the first production candidate.

No hardware measurement is authorized while there is no opportunity-linked
candidate that has passed smoke screening. A measurement is useful only when a
named uncertainty can change the ordering of working candidates.

Default budgets are twenty real wall-clock minutes per candidate, two hours
from the first registered candidate for the whole portfolio, and eight
technical repairs per candidate. Count agent reasoning and source-editing time,
not only GPU command duration. Expiry pauses discovery for a portfolio-level
decision and must not silently fall through into more measurement.
