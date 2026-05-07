# gpu-probing-cache

Prototype implementation of probing and bound propagation for MILP heuristics, focused on GPU-side changed-bound detection and sparse bound-update compaction.

## Goal

Evaluate whether detecting and compacting changed bounds directly on the GPU can reduce host-device transfer overhead during probing cache construction.

## Planned Features

- Simple probing mechanism
- Bound propagation
- Probing cache construction
- CPU baseline implementation
- GPU-side changed-bound detection
- Sparse bound-update compaction
- Benchmarking on MIP instances