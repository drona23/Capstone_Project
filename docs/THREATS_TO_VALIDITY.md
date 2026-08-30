# Threats to Validity

## External validity

The default inputs are synthetic. They demonstrate behavior under controlled
tradeoffs but cannot estimate savings for an actual fleet. Real job arrivals,
hardware constraints, carbon signals, WUE, and capacity congestion may produce
materially different results.

Only four zones and one simulated week are modeled per seed. Seasonal effects,
grid outages, heat waves, drought restrictions, and long-running workloads are
outside the benchmark.

## Construct validity

Operational carbon uses average grid intensity. Marginal emissions may be more
appropriate when the research question concerns the consequence of an
incremental load decision.

The water model includes onsite cooling and humidification represented by WUE.
It excludes electricity-generation water, supply-chain water, and withdrawal
versus consumption distinctions. Research on AI water footprints shows why those
boundaries matter [[R11](REFERENCES.md#r11)].

Scarcity-weighted water is a simple multiplier. It is not a watershed impact
assessment or environmental equity metric. The equity dimension studied by Li
et al. requires a broader distributional model [[R9](REFERENCES.md#r9)].

The relocation indicator is not measured latency, data-transfer energy, network
cost, or legal feasibility.

## Internal validity

The scheduler is greedy. Earlier assignments consume capacity that can change
later decisions. The chosen job order and tie-breaking rules therefore affect
the result. No claim of global optimality is made.

Carbon and water components are normalized within each job's current feasible
candidate set. This makes weights interpretable within a decision, but scores
are not comparable across jobs.

The random baseline is seeded and feasible, but it is not an estimate of a real
operator's policy. Immediate local is also a simplified operational reference.

## Statistical conclusion validity

Approximate confidence intervals use 10 seeds and a normal critical value. The
seeds are independent generator states but do not represent a sampled population
of real fleets. More seeds and trace-derived replications are needed for a paper.

The forecast-error process is Gaussian and independent by hour. Real forecasting
errors can be biased, autocorrelated, heavy-tailed, and synchronized across
regions. The reported stress test is diagnostic, not a calibrated forecast study.

## Measurement validity

PUE and WUE can vary with load, temperature, cooling technology, and reporting
boundary. The benchmark treats PUE as static per site and WUE as hourly input.
Microsoft's public definitions illustrate that both metrics depend on facility
and location conditions [[R15](REFERENCES.md#r15)].

Power demand is constant throughout a job. Accelerator utilization, idle power,
checkpointing, and variable resource profiles are not modeled.

## Missing production constraints

The artifact does not enforce:

- data residency and privacy;
- hardware or accelerator compatibility;
- network bandwidth and transfer time;
- storage locality;
- electricity price or demand charges;
- job dependencies and preemption;
- queue fairness across users;
- failures, maintenance, or reserve margins;
- embodied emissions and hardware lifetime.

These limitations prevent production use but create a clear agenda for follow-on
research.
