"""
fetch_watttime.py — NOT USED
-----------------------------
WattTime's /v3/historical endpoint requires a paid subscription.
The free tier only previews CAISO_NORTH (California), which covers
none of the 28 AI data center locations in this project.

Carbon intensity is instead computed in build_dataset.py using:
  - EIA hourly fuel mix (free)         ← fetch_eia.py
  - EPA eGRID 2022 emission factors    ← hardcoded constants

This is the standard methodology in peer-reviewed carbon-aware
computing research and is fully reproducible without any paid API.

If you obtain a WattTime subscription in the future, marginal MOER
would be a stronger signal than average intensity — update the
co2_kg column in build_dataset.py with WattTime values at that point.

Subscription info: https://watttime.org/solutions/
"""
