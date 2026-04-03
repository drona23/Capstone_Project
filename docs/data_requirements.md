# Data Requirements and Online Sources

## Best Dataset Available Right Now

The strongest dataset currently available for this project is the master table
inside `Archive 3.zip`:

- `base_data_with_metrics.parquet`
- `base_data_with_metrics.csv`

This dataset is already strong enough for the prediction pipeline because it
contains:

- hourly timestamps from 2019 through 2023
- city and ZIP identifiers
- eGRID regions
- weather variables
- generation mix variables
- `WUE_total`
- `liters`
- `co2_kg`
- water scarcity signal `dsci_0_500`

## What We Still Need Exactly

### 1. Workload or Job Trace Data

This is the main missing input for the scheduling stage.

Required fields:

- `job_id`
- `origin_city`
- `power_demand`
- `duration_hours`
- `earliest_start`
- `deadline`

Optional but helpful:

- `max_distance_km`
- `job_type`
- `priority`
- `latency_sla_ms`

Can we get it online:

- Yes, public workload traces exist.
- They usually do not come with city labels, so we may still need to map or
  synthesize `origin_city`.

Good sources:

- [Google cluster-data](https://github.com/google/cluster-data)
- [Alibaba clusterdata](https://github.com/alibaba/clusterdata)
- [Azure Trace for Packing 2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md)

Recommended first choice:

- Azure Trace for Packing 2020
  - Direct download:
    [AzurePackingTraceV1.zip](https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurevmallocation_dataset2020/AzurePackingTraceV1.zip)
  - Why:
    it includes VM start/end times, priorities, and normalized resource sizes,
    and the compressed download is manageable.

### 2. Data Center Inventory and Constraints

This is the second major missing input for scheduling.

Required fields:

- `dc_id`
- `city`
- `max_power_per_hour`
- `pue`
- `water_usage_factor`
- `region_water_scarcity`
- `is_new`

Can we get it online:

- Partially.
- City and region presence can be sourced online.
- Exact per-facility capacity, hourly power ceilings, and water-usage factors
  are usually not public at the level we need.
- In practice, this part often needs assumptions, proxy values, or a synthetic
  scenario table.

Useful sources:

- [AWS Global Infrastructure](https://aws.amazon.com/about-aws/global-infrastructure/)
- [Google Cloud Locations](https://cloud.google.com/about/locations)
- [Azure Global Infrastructure](https://azure.microsoft.com/en-us/explore/global-infrastructure/geographies/)
- [Google Data Center Efficiency](https://datacenters.google/efficiency/)

### 3. Water Scarcity Weights

We already have `dsci_0_500` in the master dataset, but if we want a fresh or
more explicit scarcity layer, we can source it online.

Can we get it online:

- Yes.

Good source:

- [WRI Aqueduct Water Risk Atlas](https://www.wri.org/data/aqueduct-water-risk-atlas)

### 4. Forecast Weather for Future Inference

Historical weather is already available in the master dataset, but future
forecasting requires forecast weather inputs.

Can we get it online:

- Yes.

Good source:

- [NOAA / NWS API](https://www.weather.gov/documentation/services-web-api)

### 5. City Coordinates for Forecast APIs

If we use NOAA or other live forecast services, we need city coordinates.

Can we get it online:

- Yes.

Good source:

- [U.S. Census Geocoder](https://geocoding.geo.census.gov/)

## Recommendation

The best next step is:

1. Use `base_data_with_metrics.parquet` as the primary training dataset.
2. Use the Azure packing trace as the first workload trace source for scheduling.
3. Create a small synthetic or estimated data-center configuration table.
4. Only after that, operationalize the scheduling notebook into the main codebase.

## What We Do Not Strictly Need Right Now

We do not need more historical weather or historical carbon data immediately if
we use the master dataset from `Archive 3.zip`, because those signals are
already present there.
