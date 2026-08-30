# Related Work and Project Positioning

## Summary

Carbon-aware computing and water-aware workload management are established
research areas. The defensible value of this project is its reproducible
evaluation scaffold, not a first-of-kind scheduling claim.

| Work | Main contribution | Difference from this project |
|---|---|---|
| Radovanovic et al., Carbon-Aware Computing for Datacenters [[R2](REFERENCES.md#r2)] | Risk-aware control of temporally flexible workloads across Google's fleet using forecast-driven capacity curves | This artifact is a small offline simulator. It adds explicit onsite water and paired-job metrics, but it has no fleet deployment or production validation. |
| Wiesner et al., Let's Wait Awhile [[R3](REFERENCES.md#r3)] | Quantifies temporal shifting potential and studies carbon forecast accuracy across four grids | This project combines temporal and geographic placement with site capacity and water impact. Its default data are synthetic rather than a year of historical grid data. |
| Islam et al., WACE [[R4](REFERENCES.md#r4)] | Online delay-tolerant scheduling that jointly considers water, carbon, and electricity cost | WACE predates and overlaps the core environmental objective. This project omits price and online control, while emphasizing paired evaluation and a reproducible offline artifact. |
| Acun et al., Carbon Explorer [[R5](REFERENCES.md#r5)] | Design-space analysis across renewable capacity, batteries, scheduling, operational carbon, and embodied carbon | This project studies job placement only. It does not model renewable procurement, storage, or embodied emissions. |
| Souza et al., Ecovisor [[R6](REFERENCES.md#r6)] | Application-facing abstraction for renewable energy, grid carbon, server power, and batteries | This project does not expose an application runtime or energy virtualization layer. It evaluates a scheduling policy offline. |
| Hanafy et al., CarbonScaler [[R7](REFERENCES.md#r7)] | Dynamically changes resource allocation for elastic batch workloads to improve carbon efficiency | This project assumes fixed job power and non-preemptive execution. It shifts start time and location instead of scaling resource allocation. |
| Qi et al., SHIELD [[R8](REFERENCES.md#r8)] | Multi-objective geo-distributed optimization of carbon, wastewater, and energy cost using evolutionary search and learned guidance | SHIELD is the closest objective-level comparison. This project is simpler and more interpretable, adds deadline and hourly capacity checks, and avoids claiming Pareto superiority. |
| Li et al., environmentally equitable AI [[R9](REFERENCES.md#r9)] | Carbon and water aware geographic load balancing with an environmental equity objective | This project uses a scarcity-weighted water metric but does not model distributional equity across communities. Equity is a future extension, not a current result. |
| Xu et al., GREEN [[R10](REFERENCES.md#r10)] | Carbon-efficient ML cluster scheduler with job-completion-time evaluation and production workload traces | This project adds geographic and water dimensions, but lacks a cluster prototype, detailed JCT model, and production trace validation. |
| Li et al., Making AI Less Thirsty [[R11](REFERENCES.md#r11)] | Method and policy discussion for the spatial and temporal water footprint of AI | This project operationalizes onsite WUE in a scheduler, but it excludes electricity-generation water and should not be read as a full AI water-footprint model. |

## Claim boundary

The project can reasonably claim that it implements and tests a transparent
combination of known ideas under a consistent evaluation protocol. It should not
claim any of the following:

- the first carbon-aware scheduler;
- the first water-aware scheduler;
- the first joint carbon-water scheduler;
- production readiness;
- superiority to evolutionary, optimization, or deployed systems;
- real-world savings based on the synthetic benchmark.

## Research gap this artifact can support

A stronger future paper could investigate whether paired functional-unit
evaluation, forecast uncertainty, deadline slack, and regional water stress
change the conclusions drawn from conventional carbon-only baselines. That study
would need trace-derived jobs, measured or operator-reported WUE, authoritative
hourly carbon data, network and policy constraints, and a larger experimental
design.
