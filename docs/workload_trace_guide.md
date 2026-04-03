# Workload Trace Guide

## Recommended Public Trace

Use the official Microsoft Azure packing trace:

- Dataset doc:
  [Azure Trace for Packing 2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md)
- Direct download:
  [AzurePackingTraceV1.zip](https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurevmallocation_dataset2020/AzurePackingTraceV1.zip)

## Why This One

This trace is a good fit for the scheduling stage because it contains:

- VM request ids
- tenant ids
- VM types
- VM priority
- start times
- end times
- normalized resource requirements through the `vmType` table

The download is practical at roughly 51 MB compressed, and it expands into a
single SQLite file with millions of VM requests.

## What We Downloaded Locally

Local path used during setup:

- `data/external/AzurePackingTraceV1.zip`
- `data/external/azure_packing/packing_trace_zone_a_v1.sqlite`

These files are intentionally ignored by Git because they are raw external
artifacts.

## Schema We Found

The SQLite database contains two tables:

- `vm`
- `vmType`

The `vm` table provides:

- `vmId`
- `tenantId`
- `vmTypeId`
- `priority`
- `starttime`
- `endtime`

The `vmType` table provides:

- `vmTypeId`
- `machineId`
- `core`
- `memory`
- `hdd`
- `ssd`
- `nic`

## How We Use It

The project now includes [src/workload_loader.py](../src/workload_loader.py),
which converts the Azure trace into a scheduler-oriented jobs table.

Example:

```bash
python3 -m src.workload_loader \
  --input-path data/external/azure_packing/packing_trace_zone_a_v1.sqlite \
  --output-path data/processed/azure_jobs_sample.csv \
  --limit 5000
```

## Important Limitations

This trace is excellent for workload behavior, but it does not directly include:

- real `origin_city`
- explicit `power_demand` in kW
- user-facing latency SLAs
- true geographic placement

Because of that, the current converter uses proxies:

- `origin_city` is deterministically assigned from tenant id across a city pool
- `power_demand` is approximated from normalized CPU allocation
- `deadline` is derived from observed duration plus configurable slack

These assumptions are acceptable for a research scheduler prototype, but they
should be stated clearly in any report or presentation.
