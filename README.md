# amdgpu-virtual-packages

A small conda plugin providing AMD GPU virtual packages:

- `__amdgpu=0=0` when an AMD GPU is available.
- `__amdgpu_arch=<arch>=0` with the detected GPU architecture, for example `gfx1151` as `11.5.1`.

It supports Linux, Windows, and Linux under WSL. If multiple AMD GPUs are detected,
`__amdgpu_arch` reports the architecture of the GPU with the most compute units; architecture
ordering breaks compute-unit-count ties.

To inspect the detected virtual packages locally with [Pixi](https://pixi.sh):

```console
pixi run print-amdgpu-virtual-packages
```

Example output:

```text
__amdgpu=0=0
__amdgpu_arch=11.5.1=0
selected_gpu_compute_units=16
```

The compute-unit count is reported as `unknown` when the selected GPU's architecture can be
detected but its compute-unit count cannot.

Related issues and PRs:
* https://github.com/conda/ceps/pull/189
* https://github.com/gbionics/rock-the-conda/issues/4
