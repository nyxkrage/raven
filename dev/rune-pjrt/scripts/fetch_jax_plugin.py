#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Copyright (c) 2026 The Raven authors. All rights reserved.
# SPDX-License-Identifier: ISC
# ---------------------------------------------------------------------------

"""Locate or fetch the PJRT CUDA plugin distributed by JAX."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable


PYPI_JSON = "https://pypi.org/pypi/{package}/json"
USER_AGENT = "raven-rune-pjrt/1"
PLUGIN_NAME = "xla_cuda_plugin.so"


class Fetch_error(RuntimeError):
    pass


def command_output(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            arguments,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def cuda_from_driver() -> int | None:
    output = command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if not output:
        return None
    match = re.match(r"\s*(\d+)", output)
    if match is None:
        return None
    major = int(match.group(1))
    if major >= 580:
        return 13
    return 12 if major >= 525 else None


def cuda_from_nvcc() -> int | None:
    output = command_output(["nvcc", "--version"])
    if not output:
        return None
    match = re.search(r"\brelease\s+(\d+)", output)
    return int(match.group(1)) if match else None


def cuda_preference(requested: str) -> tuple[list[int], bool]:
    if requested in ("12", "13"):
        return ([int(requested)], True)
    detected = cuda_from_driver()
    if detected is None:
        detected = cuda_from_nvcc()
    if detected is None:
        return ([13, 12], False)
    major = 13 if detected >= 13 else 12
    return ([major, 12 if major == 13 else 13], True)


def plugin_relative_path(cuda: int) -> Path:
    return Path("jax_plugins") / f"xla_cuda{cuda}" / PLUGIN_NAME


def site_roots() -> list[Path]:
    roots = [Path(path) for path in sys.path if path]
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except AttributeError:
        pass
    roots.append(Path(site.getusersitepackages()))
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = os.path.realpath(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def installed_plugin(cuda_order: Iterable[int]) -> Path | None:
    for cuda in cuda_order:
        relative = plugin_relative_path(cuda)
        for root in site_roots():
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    return None


def default_cache_dir() -> Path:
    configured = os.environ.get("RUNE_PJRT_PLUGIN_CACHE")
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "raven" / "rune-pjrt"


def version_key(version: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"([0-9]+)", version):
        if not part:
            continue
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    return tuple(parts)


def cached_plugin(cache_dir: Path, cuda_order: Iterable[int]) -> Path | None:
    for cuda in cuda_order:
        package_dir = cache_dir / f"jax-cuda{cuda}-pjrt"
        if not package_dir.is_dir():
            continue
        versions = sorted(
            (path for path in package_dir.iterdir() if path.is_dir()),
            key=lambda path: version_key(path.name),
            reverse=True,
        )
        relative = plugin_relative_path(cuda)
        for version in versions:
            candidate = version / relative
            if candidate.is_file():
                return candidate.resolve()
    return None


def load_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except (OSError, ValueError) as error:
        raise Fetch_error(f"failed to read {url}: {error}") from error


def platform_architecture() -> str:
    if sysconfig.get_platform().split("-")[0] != "linux":
        raise Fetch_error("JAX PJRT CUDA wheels are available only on Linux")
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    raise Fetch_error(f"unsupported machine architecture {machine!r}")


def wheel_metadata(cuda: int, requested_version: str | None) -> dict[str, str]:
    package = f"jax-cuda{cuda}-pjrt"
    metadata = load_json(PYPI_JSON.format(package=package))
    version = requested_version or str(metadata["info"]["version"])
    if version == metadata["info"]["version"]:
        files = metadata["urls"]
    else:
        release = load_json(
            PYPI_JSON.format(package=f"{package}/{version}")
        )
        files = release["urls"]
    architecture = platform_architecture()
    suffix = f"_{architecture}.whl"
    candidates = [
        file
        for file in files
        if file.get("packagetype") == "bdist_wheel"
        and not file.get("yanked", False)
        and str(file["filename"]).endswith(suffix)
        and "-py3-none-manylinux" in str(file["filename"])
    ]
    if not candidates:
        raise Fetch_error(
            f"{package} {version} has no compatible wheel for {architecture}"
        )
    file = candidates[0]
    return {
        "package": package,
        "version": version,
        "filename": str(file["filename"]),
        "url": str(file["url"]),
        "sha256": str(file["digests"]["sha256"]),
    }


def download_wheel(metadata: dict[str, str], destination: Path) -> None:
    request = urllib.request.Request(
        metadata["url"], headers={"User-Agent": USER_AGENT}
    )
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with destination.open("wb") as output:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise Fetch_error(f"failed to download {metadata['url']}: {error}") from error
    actual = digest.hexdigest()
    if actual != metadata["sha256"]:
        destination.unlink(missing_ok=True)
        raise Fetch_error(
            f"SHA256 mismatch for {metadata['filename']}: "
            f"expected {metadata['sha256']}, got {actual}"
        )


def extract_plugin(
    wheel: Path, release_dir: Path, cuda: int, metadata: dict[str, str]
) -> Path:
    relative = plugin_relative_path(cuda)
    destination = release_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".so.part")
    try:
        with zipfile.ZipFile(wheel) as archive:
            member = relative.as_posix()
            try:
                info = archive.getinfo(member)
            except KeyError as error:
                raise Fetch_error(
                    f"{metadata['filename']} does not contain {member}"
                ) from error
            with archive.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
        temporary.chmod(0o755)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = release_dir / "wheel.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return destination.resolve()


def fetch_plugin(
    cache_dir: Path, cuda: int, requested_version: str | None
) -> Path:
    metadata = wheel_metadata(cuda, requested_version)
    release_dir = cache_dir / metadata["package"] / metadata["version"]
    destination = release_dir / plugin_relative_path(cuda)
    if destination.is_file():
        return destination.resolve()
    release_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=metadata["filename"] + ".",
        suffix=".part",
        dir=release_dir,
        delete=False,
    ) as temporary:
        wheel = Path(temporary.name)
    try:
        print(
            f"rune-pjrt: downloading {metadata['package']} "
            f"{metadata['version']} from PyPI",
            file=sys.stderr,
        )
        download_wheel(metadata, wheel)
        return extract_plugin(wheel, release_dir, cuda, metadata)
    finally:
        wheel.unlink(missing_ok=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        choices=("auto", "12", "13"),
        default=os.environ.get("RUNE_PJRT_CUDA_VERSION", "auto"),
        help="CUDA wheel family (default: detect from driver/toolkit)",
    )
    parser.add_argument(
        "--version",
        default=os.environ.get("RUNE_PJRT_JAX_VERSION"),
        help="JAX PJRT wheel version (default: latest stable PyPI release)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="plugin cache directory",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="only search installed wheels and the existing cache",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    cache_dir = arguments.cache_dir.expanduser()
    cuda_order, can_download = cuda_preference(arguments.cuda)
    found = installed_plugin(cuda_order)
    if found is None:
        found = cached_plugin(cache_dir, cuda_order)
    if found is not None:
        print(found)
        return 0
    if arguments.no_download:
        raise Fetch_error("no installed or cached JAX PJRT CUDA plugin was found")
    if not can_download:
        raise Fetch_error(
            "could not detect an NVIDIA driver or CUDA toolkit; "
            "set RUNE_PJRT_CUDA_VERSION=12 or 13 to select a wheel"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".fetch.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        found = cached_plugin(cache_dir, cuda_order)
        if found is None:
            found = fetch_plugin(cache_dir, cuda_order[0], arguments.version)
    print(found)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fetch_error as error:
        print(f"rune-pjrt-fetch-plugin: {error}", file=sys.stderr)
        raise SystemExit(1)
