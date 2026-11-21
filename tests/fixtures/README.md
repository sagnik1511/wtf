# Compatibility fixtures

These small configurations freeze legacy observation and action contracts for
mating, inheritance, recording and checkpoint regression tests. They are test
inputs, not recommended simulation presets or archived experiment results.

- `mating_v3.json`: version-3 mating observations and nearest-only predator inputs.
- `consent_v4_nearest.json`: version-4 consent inputs with nearest-only predators.

Current user-facing settings belong in `environments/`. Keep compatibility tests
independent of superseded local presets when updating those maintained defaults.
