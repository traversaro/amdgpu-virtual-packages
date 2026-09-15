import ctypes
import ctypes.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from conda import plugins


_KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")
_DRM = Path("/sys/class/drm")
_DLL_DIRECTORY_HANDLES = []


def _base36_digit(value: int) -> str:
    if 0 <= value < 10:
        return str(value)
    if 10 <= value < 36:
        return chr(ord("a") + value - 10)
    raise ValueError(f"AMDGPU target component out of range: {value}")


def _gfx_from_kfd_target(target: int) -> str:
    major = target // 10000
    minor = (target // 100) % 100
    stepping = target % 100
    return f"gfx{major}{_base36_digit(minor)}{_base36_digit(stepping)}"


def _gfx_key(gfx: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"gfx([0-9]+)([0-9a-z])([0-9a-z])", gfx)
    if match is None:
        raise ValueError(f"Invalid AMDGPU architecture: {gfx}")

    def value(char: str) -> int:
        return int(char, 36)

    return int(match.group(1)), value(match.group(2)), value(match.group(3))


def _gfx_version(gfx: str) -> str:
    major, minor, stepping = _gfx_key(gfx)
    return f"{major}.{minor}.{stepping}"


def _kfd_devices() -> list[tuple[str, int | None]]:
    devices: list[tuple[str, int | None]] = []
    if not _KFD_TOPOLOGY.is_dir():
        return devices

    for properties in _KFD_TOPOLOGY.glob("*/properties"):
        try:
            text = properties.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"^gfx_target_version\s+(\d+)\s*$", text, re.MULTILINE)
        if match is None:
            continue
        target = int(match.group(1))
        if target == 0:
            continue
        try:
            gfx = _gfx_from_kfd_target(target)
        except ValueError:
            continue

        simd_count = re.search(r"^simd_count\s+(\d+)\s*$", text, re.MULTILINE)
        simd_per_cu = re.search(r"^simd_per_cu\s+(\d+)\s*$", text, re.MULTILINE)
        if simd_count is None or simd_per_cu is None or int(simd_per_cu.group(1)) == 0:
            compute_units = None
        else:
            compute_units = int(simd_count.group(1)) // int(simd_per_cu.group(1))
        devices.append((gfx, compute_units))
    return devices


def _kfd_architectures() -> set[str]:
    return {gfx for gfx, _ in _kfd_devices()}


def _linux_has_amdgpu() -> bool:
    if not _DRM.is_dir():
        return False

    for card in _DRM.iterdir():
        if re.fullmatch(r"card\d+", card.name) is None:
            continue
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text(encoding="ascii").strip().lower()
            driver = (device / "driver").resolve().name
        except OSError:
            continue
        if vendor == "0x1002" and driver == "amdgpu":
            return True
    return False


def _is_wsl() -> bool:
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="ascii")
    except OSError:
        return False
    return "microsoft" in release.lower()


def _windows_host_has_amdgpu() -> bool:
    if os.name != "nt" and not _is_wsl():
        return False
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        return False
    command = (
        "$g = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | "
        "Where-Object { $_.PNPDeviceID -match 'VEN_1002' }); "
        "if ($g.Count -gt 0) { exit 0 } else { exit 1 }"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _hip_library_candidates() -> list[str]:
    candidates: list[str] = []

    if os.name == "nt":
        directories = [
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.executable).parent,
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
            Path.cwd(),
        ]
        directories.extend(Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry)

        versioned: list[Path] = []
        for directory in directories:
            try:
                versioned.extend(directory.glob("amdhip64_*.dll"))
            except OSError:
                continue

        def dll_version(path: Path) -> int:
            match = re.fullmatch(r"amdhip64_(\d+)\.dll", path.name, re.IGNORECASE)
            return int(match.group(1)) if match else -1

        candidates.extend(str(path) for path in sorted(versioned, key=dll_version, reverse=True))
        candidates.extend(str(directory / "amdhip64.dll") for directory in directories)
        candidates.append("amdhip64.dll")
    else:
        found = ctypes.util.find_library("amdhip64")
        if found:
            candidates.append(found)
        candidates.extend(
            [
                str(Path(sys.prefix) / "lib" / "libamdhip64.so"),
                "/opt/rocm/lib/libamdhip64.so",
                "/opt/rocm/lib64/libamdhip64.so",
                "libamdhip64.so",
            ]
        )

    return list(dict.fromkeys(candidates))


def _load_hip_runtime():
    for candidate in _hip_library_candidates():
        path = Path(candidate)
        try:
            if os.name == "nt" and path.is_absolute() and path.parent.is_dir():
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path.parent)))
            return ctypes.CDLL(candidate)
        except (OSError, FileNotFoundError):
            continue
    return None


def _hip_devices() -> list[tuple[str, int | None]]:
    library = _load_hip_runtime()
    if library is None:
        return []

    try:
        get_count = library.hipGetDeviceCount
    except AttributeError:
        return []

    get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_count.restype = ctypes.c_int
    count = ctypes.c_int()
    if get_count(ctypes.byref(count)) != 0 or count.value <= 0:
        return []

    property_apis = []
    for symbol, offset in (
        ("hipGetDevicePropertiesR0600", 1160),
        ("hipGetDevicePropertiesR0000", 396),
        ("hipGetDeviceProperties", 396),
    ):
        try:
            function = getattr(library, symbol)
        except AttributeError:
            continue
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_int
        property_apis.append((function, offset))

    try:
        get_attribute = library.hipDeviceGetAttribute
    except AttributeError:
        get_attribute = None
    else:
        get_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        get_attribute.restype = ctypes.c_int

    devices: list[tuple[str, int | None]] = []
    for device in range(count.value):
        for get_properties, offset in property_apis:
            buffer = ctypes.create_string_buffer(4096)
            if get_properties(buffer, device) != 0:
                continue
            raw = bytes(buffer[offset : offset + 256]).split(b"\0", 1)[0]
            try:
                gfx = raw.decode("ascii").split(":", 1)[0]
                _gfx_key(gfx)
            except (UnicodeDecodeError, ValueError):
                continue
            compute_units = None
            if get_attribute is not None:
                value = ctypes.c_int()
                # hipDeviceAttributeMultiprocessorCount has the stable value 16.
                if get_attribute(ctypes.byref(value), 16, device) == 0:
                    compute_units = value.value
            devices.append((gfx, compute_units))
            break
    return devices


def _hip_architectures() -> set[str]:
    return {gfx for gfx, _ in _hip_devices()}


def _devices() -> list[tuple[str, int | None]]:
    devices = _kfd_devices()
    if not devices:
        devices = _hip_devices()
    return devices


def _selected_device(devices: list[tuple[str, int | None]]) -> tuple[str, int | None]:
    return max(
        devices,
        key=lambda device: (device[1] if device[1] is not None else -1, _gfx_key(device[0])),
    )


def _selected_architecture(devices: list[tuple[str, int | None]]) -> str:
    return _selected_device(devices)[0]


def virtual_packages(
    devices: list[tuple[str, int | None]] | None = None,
) -> list[tuple[str, str, str]]:
    if devices is None:
        devices = _devices()

    has_amdgpu = bool(devices)
    if os.name == "posix":
        has_amdgpu = has_amdgpu or _linux_has_amdgpu()
    has_amdgpu = has_amdgpu or _windows_host_has_amdgpu()

    if not has_amdgpu:
        return []

    packages = [("amdgpu", "0", "0")]
    if devices:
        selected = _selected_architecture(devices)
        packages.append(("amdgpu_arch", _gfx_version(selected), "0"))
    return packages


@plugins.hookimpl
def conda_virtual_packages():
    for name, version, build in virtual_packages():
        yield plugins.CondaVirtualPackage(name=name, version=version, build=build)
