# Data Provenance and Schemas

## Default experiment

The default experiment uses no downloaded data. `src/synthetic_benchmark.py`
generates every input deterministically from a seed. The four zone names are
fictional. Signal levels, site characteristics, and jobs are plausible test
values created for method development. They are not measurements and do not
represent named companies, facilities, grids, or communities.

| Item | Provenance | Version control status |
|---|---|---|
| Hourly carbon and water signals | Deterministic synthetic generator | Source code and generated aggregate results are tracked |
| Job arrivals, duration, power, slack, origin, priority | Deterministic synthetic generator | Source code is tracked |
| PUE, capacity, water factor, scarcity | Deterministic synthetic configuration | Source code is tracked |
| Result tables and figures | `python -m src.experiments` | Tracked |

## Canonical environment schema

| Column | Unit | Meaning |
|---|---|---|
| `timestamp` | UTC hour | Start of the environmental observation |
| `city` | text | Modeled execution zone |
| `carbon_intensity_kg_per_kwh` | kg CO2e per facility kWh | Average operational electricity intensity used by the experiment |
| `water_intensity_l_per_kwh` | liters per IT kWh | Onsite WUE for cooling and humidification |

## Canonical job schema

| Column | Unit | Meaning |
|---|---|---|
| `job_id` | text | Unique functional unit identifier |
| `origin_city` | text | Preferred or originating execution zone |
| `power_demand_kw` | kW | Constant IT power during execution |
| `duration_hours` | hours | Positive integer, non-preemptive duration |
| `earliest_start` | UTC timestamp | Release time |
| `deadline` | UTC timestamp | Required completion time |
| `priority` | ordinal integer | Lower values are considered first |

## Canonical data center schema

| Column | Unit | Meaning |
|---|---|---|
| `dc_id` | text | Unique site identifier |
| `city` | text | Join key for environmental signals |
| `max_power_kw` | kW | Hourly IT power capacity |
| `pue` | ratio | Facility energy divided by IT energy |
| `water_usage_factor` | multiplier | Site-specific adjustment to the input WUE |
| `water_scarcity_index` | 0 to 1 | Relative local scarcity used in the impact score |

## Optional external adapters

The adapters are not used by the committed benchmark. They are starting points
for future trace-based research and require an explicit provenance record for
every download.

| Adapter | Primary source | Output | Important boundary |
|---|---|---|---|
| `src/fetch_eia.py` | U.S. Energy Information Administration Open Data API [[R12](REFERENCES.md#r12)] | Hourly generation mix by balancing authority | Generation mix is not itself a validated marginal carbon signal |
| `src/fetch_entso_e.py` | ENTSO-E Transparency Platform [[R13](REFERENCES.md#r13)] | Hourly energy by production type after interval aggregation | Access requires registration and remains subject to platform terms |
| `src/workload_loader.py` | Microsoft Azure public VM traces [[R14](REFERENCES.md#r14)] | Scheduler job table | CPU allocation is converted to a power proxy and tenant is mapped to a synthetic origin |

The Azure repository identifies its released trace data under CC BY 4.0 and its
code under MIT. Check the source repository at download time. EIA and ENTSO-E
users must also review the current source terms and citation requirements.

## Required provenance record for a real-data study

For each file, record:

- source organization and direct URL;
- dataset or API version;
- retrieval timestamp in UTC;
- query parameters and geographic identifiers;
- license or terms of use;
- checksum of the raw file;
- timezone and unit conversions;
- missing-data and aggregation rules;
- whether carbon is average or marginal;
- whether water includes onsite consumption, offsite electricity water, or both.

The source pages in this document were checked on 2026-08-30.
