from amdgpu_virtual_packages.plugin import _devices, _selected_device, virtual_packages

devices = _devices()

for name, version, build in virtual_packages(devices):
    print(f"__{name}={version}={build}")

if devices:
    _, compute_units = _selected_device(devices)
    value = compute_units if compute_units is not None else "unknown"
    print(f"selected_gpu_compute_units={value}")
