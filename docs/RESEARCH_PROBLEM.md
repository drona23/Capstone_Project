# Research Problem and Use Cases

## Business problem

Compute operators usually optimize placement for service quality, utilization,
and cost. Environmental signals introduce another operational question: if a job
can wait and more than one site can execute it, which feasible site and start time
produce the preferred carbon-water tradeoff?

This is not a recommendation to delay interactive requests or move regulated data
across borders. The scoped workload is non-preemptive, deadline-flexible compute,
such as model training, batch inference, scientific workflows, data engineering,
CI jobs, report generation, backups, and software maintenance. Temporal and
geographic shifting of flexible workloads is also recognized in published systems
research and cloud architecture guidance [[R2](REFERENCES.md#r2),
[R3](REFERENCES.md#r3), [R17](REFERENCES.md#r17),
[R18](REFERENCES.md#r18)].

## Decision statement

Given hourly forecasts, a set of jobs, and a portfolio of data centers, select one
site and one contiguous execution window for each admitted job so that:

1. execution does not start before release time;
2. execution ends no later than the deadline;
3. IT power does not exceed site capacity in any hour;
4. environmental impact and relocation preference are evaluated consistently;
5. unscheduled jobs remain visible in the reported coverage metric.

## Primary research question

How much do spatiotemporal placement decisions change operational carbon,
physical onsite water use, and scarcity-weighted water impact relative to
feasible operational baselines, at equal completed work?

## Secondary questions

- How does a balanced carbon-water policy compare with a carbon-only policy?
- How sensitive is a carbon-only policy to forecast error?
- Do claimed reductions persist across workload and signal seeds?
- What fraction of submitted jobs remains feasible under each policy?

## Organizations that could benefit

### Cloud and AI infrastructure operators

AWS, Google Cloud, Microsoft Azure, and other multi-region operators are
representative of organizations with placement flexibility. Cloud guidance
already treats region choice and carbon-aware timing as sustainability decisions
[[R17](REFERENCES.md#r17), [R18](REFERENCES.md#r18)]. A production version of this
research could inform a batch admission controller or planning service.

### Universities, national laboratories, and HPC centers

Research institutions operate queued workloads with explicit resource requests
and completion expectations. They can use this artifact to study how much
deadline slack is needed before environmental signals affect scheduling, without
changing a production scheduler.

### Colocation and enterprise data center portfolios

Operators with several sites can use the experiment design for scenario analysis.
The most relevant decision is whether carbon and water objectives align or
conflict across facilities.

### Utilities, grid operators, and public agencies

Flexible compute can be studied as a form of load flexibility. This repository
can support controlled simulations that connect hourly grid signals to admitted
IT load, but it does not yet model grid dispatch or demand response payments.

## Success criteria

A useful research result must report all of the following:

- deadline and capacity feasibility;
- coverage for every policy;
- the number and IT energy of jobs completed by both policies;
- operational carbon in kg CO2e;
- physical onsite water in liters;
- scarcity-weighted water as a separate decision metric;
- uncertainty across seeds;
- sensitivity to imperfect forecasts;
- limitations that prevent direct operational claims.

## Non-goals

- A production orchestrator or cloud broker
- An interactive dashboard
- A real-time API
- A claim that carbon-aware or water-aware scheduling is novel
- A claim that synthetic results predict savings for a named company
- A complete life-cycle assessment of computing
