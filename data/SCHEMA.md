# Telemetry adapter schema

The benchmark consumes a long-form CSV or DataFrame with these required
columns:

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | ISO-8601 or numeric | Observation time, sorted within each asset/channel |
| `asset_id` | string | Spacecraft or subsystem identity |
| `channel_id` | string | Telemetry channel identity |
| `value` | numeric | Observed telemetry value; missing values remain missing |

Optional columns:

| Column | Meaning |
|---|---|
| `mode` | Operating-mode label available at the prediction timestamp |
| `command` | Command or command-context feature available at the prediction timestamp |
| `mission_id` | Mission or domain identity used for complete-domain holdout |
| `event_id` | Event grouping key; all windows from one event must stay together |
| `anomaly_start` | Explicit event onset timestamp or boolean onset marker |
| `anomaly_end` | Explicit event end timestamp |

Do not infer degradation, failure, or remaining useful life from an anomaly
interval alone. The adapter must document label provenance, event chronology,
interventions, censoring, cadence, missingness, license, and whether a mission-
held-out split is possible. For physical alert rates, also document the
monitored-exposure rule and the maximum gap that is treated as unmonitored.
