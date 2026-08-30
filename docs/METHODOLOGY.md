# Methodology

## System boundary

The modeled unit is one non-preemptive compute job. Each job has an origin,
IT power demand, integer duration, release time, deadline, and priority. Each data
center has an hourly IT power limit, PUE, water-use factor, and water-scarcity
index. Environmental inputs provide hourly carbon intensity and onsite Water
Usage Effectiveness for each modeled zone.

All timestamps are converted to UTC and stored without a timezone after
conversion. Hourly windows use a closed start and open end. A two-hour job that
starts at 01:00 occupies the 01:00 and 02:00 slots and ends at 03:00.

## Physical accounting

For job j at data center d over execution window H:

```text
IT energy (kWh) = power demand (kW) * duration (h)

Operational CO2e (kg) = power demand * PUE_d
                        * sum over H of carbon intensity (kg CO2e/kWh)

Onsite water (L) = power demand
                   * sum over H of WUE (L/IT kWh)
                   * site water-use factor

Scarcity-weighted water (impact liters) = onsite water
                                          * (1 + scarcity index)
```

PUE is applied to electricity and carbon. It is not applied to WUE because WUE is
defined relative to IT energy [[R15](REFERENCES.md#r15)]. Scarcity-weighted water
is an impact score, not a claim about additional physical liters consumed.

## Feasibility constraints

A candidate placement is feasible only when:

```text
start >= earliest_start
start + duration <= deadline
IT power already reserved + job power <= site max power in every occupied hour
carbon and water signals exist for every occupied hour
```

No fallback mean fills missing execution hours. Missing realized data is treated
as an evaluation error because silent imputation would weaken traceability.

## Multi-objective score

Raw carbon and water values use different units and scales. Before combining
them, candidate impacts are normalized within the feasible candidate set for the
current job:

```text
normalized(x) = (x - minimum feasible x) / (maximum feasible x - minimum feasible x)
```

If every candidate has the same value, the normalized component is zero. The
default score is:

```text
score = 0.55 * normalized operational CO2e
      + 0.35 * normalized scarcity-weighted water
      + 0.10 * relocation indicator
```

The weights are non-negative and must sum to one. The relocation indicator is 0
for the origin zone and 1 for another zone. It is a transparent proxy, not a
network-latency model.

Jobs are considered in priority, release-time, and job-ID order. The proposed
policy greedily reserves the lowest-scoring feasible candidate. This is a
heuristic, not a proof of global optimality.

## Baselines

All baselines share the same feasibility checks and job order.

| Baseline | Rule |
|---|---|
| Immediate local | Prefer the origin zone, then choose the earliest feasible slot |
| Carbon greedy | Choose the feasible candidate with the least predicted operational CO2e |
| Random feasible | Choose one feasible candidate using a seeded random generator |

The carbon-greedy baseline is intentionally strong for its single objective. A
balanced policy is expected to trade some carbon performance for water impact in
cases where those objectives conflict.

## Paired evaluation

Environmental reductions are calculated only for job IDs completed by both the
baseline and proposed policy. Coverage is reported separately for each policy.
The common job count and common IT energy are saved with every comparison. This
prevents a policy from appearing cleaner only because it completed less work.
The approach is consistent with the requirement that a sustainability comparison
keep the functional unit and methodology unchanged [[R16](REFERENCES.md#r16)].

## Synthetic benchmark

The default benchmark has:

- four fictional zones;
- 168 hourly observations per zone;
- 180 jobs per seed;
- heterogeneous power, duration, slack, origin, and priority;
- explicit carbon-water tradeoffs;
- 10 workload and signal seeds beginning at seed 42.

The signals are plausible test values, not reconstructions of real facilities.
Their purpose is repeatability, regression testing, and transparent method
development.

## Uncertainty reporting

Tables report the arithmetic mean across seeds and an approximate 95% confidence
interval half-width:

```text
CI half-width = 1.96 * sample standard deviation / square root of seed count
```

Ten seeds support a useful reproducibility check but not a population-level
claim. A publication study should use more seeds and trace-derived workloads.

## Forecast stress test

The robustness experiment isolates carbon scheduling by setting the carbon
weight to 1 and the other weights to 0. Gaussian error with zero mean is added to
each hourly carbon intensity. Its standard deviation is the configured fraction
of that city's mean intensity. Negative forecasts are clipped to zero.

The noisy policy and perfect-forecast reference are evaluated on realized carbon
for every occupied hour, including PUE. Only jobs completed by both are compared.
The reported quantity is a delta against the perfect-forecast policy, not global
optimization regret.
