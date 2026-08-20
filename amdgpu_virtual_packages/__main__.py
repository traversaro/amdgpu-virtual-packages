from amdgpu_virtual_packages.plugin import virtual_packages

for name, version, build in virtual_packages():
    print(f"__{name}={version}={build}")
