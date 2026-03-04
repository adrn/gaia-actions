# Gaia actions

## Output flag values

The `flags` column in the output files is a bitmask that encodes the processing status
for each source. A value of **0** means the source was processed successfully with no
issues.

| Value | Bit | Meaning |
|-------|-----|---------|
| `-1` | — | Source has not been processed yet |
| `0` | — | Success — no issues |
| `1` | 0 | Failed to compute valid Galactocentric positions/velocities (bad or non-finite `xyz`/`vxyz`) |
| `2` | 1 | Failed to compute actions/angles/frequencies via `agama.ActionFinder` |
| `4` | 2 | Failed to integrate the orbit |
| `8` | 3 | Failed to compute orbital parameters (`z_max`, `r_per`, `r_apo`, `ecc`) |
| `16` | 4 | Failed to compute energy (`E`) and angular momentum (`L`) |
