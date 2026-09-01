"""Build the bounded Phase A binding around unchanged QuaRot W4A4 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

QUAROT_COMMIT = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
CUTLASS_COMMIT = "ffa34e70756b0bc744e1dfcc115b5a991a68f132"
SOURCE_SHA256 = {
    "quarot/kernels/gemm.cu": "a351529098342c4e92d2a773ea33a08436e03b4bffb5efdd84b7542d95d7eabd",
    "quarot/kernels/quant.cu": "6f50ba4127380713e386b4bb2a7f78b936405db801e0eac19b1d6870f91e967e",
    "quarot/kernels/include/common.h": "7698f99d691fc342353b86896e36ed3efe7bb1fd2332d887f55fe9dd6b7ee91d",
    "quarot/kernels/include/int4.h": "02a8334f25a18eade476fe740520edd510c562f9a4c60c42bcd26e7f4cc7b22d",
    "quarot/kernels/include/gemm.h": "790e7cd9ddeb92752f959604c44957d36c7539be509aa1fd843beaaa2dac9bdc",
    "quarot/kernels/include/quant.h": "9826d20b9a058a03cecd98fbafb437c632f42892005e228d6efd4d836eb576e0",
}


class BuildRefusal(RuntimeError):
    """The pinned native dependency or build environment is not exact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise BuildRefusal(f"cannot identify Git HEAD at {root}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def verify_quarot(root: Path) -> dict[str, str]:
    root = root.resolve(strict=True)
    if _git_head(root) != QUAROT_COMMIT:
        raise BuildRefusal(f"QuaRot must be exact commit {QUAROT_COMMIT}")
    cutlass = root / "third-party" / "cutlass"
    if _git_head(cutlass) != CUTLASS_COMMIT:
        raise BuildRefusal(f"CUTLASS must be exact commit {CUTLASS_COMMIT}")
    observed = {}
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        actual = _sha256(path)
        if actual != expected:
            raise BuildRefusal(f"source digest mismatch for {relative}: {actual}")
        observed[relative] = actual
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarot-root", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_hashes = verify_quarot(args.quarot_root)
    cuda_home = args.cuda_home.resolve(strict=True)
    nvcc = cuda_home / "bin" / "nvcc"
    target_include = cuda_home / "targets" / "x86_64-linux" / "include"
    target_library = cuda_home / "targets" / "x86_64-linux" / "lib"
    if not nvcc.is_file() or not target_include.is_dir() or not target_library.is_dir():
        raise BuildRefusal("CUDA home lacks the pinned compiler target layout")
    build_directory = args.build_directory.resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    if any(build_directory.iterdir()):
        raise BuildRefusal("build directory must be empty")

    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["CPATH"] = str(target_include)
    os.environ["LIBRARY_PATH"] = str(target_library)
    os.environ["MAX_JOBS"] = "1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

    import torch
    from torch.utils.cpp_extension import load

    quarot_root = args.quarot_root.resolve(strict=True)
    binding = Path(__file__).with_name("w4a4_bindings.cpp").resolve(strict=True)
    module = load(
        name="phase_a_w4a4_cuda",
        sources=[
            str(binding),
            str(quarot_root / "quarot" / "kernels" / "gemm.cu"),
            str(quarot_root / "quarot" / "kernels" / "quant.cu"),
        ],
        extra_include_paths=[
            str(quarot_root / "quarot" / "kernels" / "include"),
            str(quarot_root / "third-party" / "cutlass" / "include"),
            str(quarot_root / "third-party" / "cutlass" / "tools" / "util" / "include"),
        ],
        extra_cflags=["-O0"],
        extra_cuda_cflags=[
            "-O2",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_86,code=sm_86",
        ],
        build_directory=str(build_directory),
        with_cuda=True,
        is_python_module=True,
        verbose=True,
    )
    extension = Path(module.__file__).resolve(strict=True)
    result = {
        "schema_version": "phase-a-w4a4-build-v1",
        "quarot_commit": QUAROT_COMMIT,
        "cutlass_commit": CUTLASS_COMMIT,
        "source_sha256": source_hashes,
        "binding_path": str(binding),
        "binding_sha256": _sha256(binding),
        "extension_path": str(extension),
        "extension_sha256": _sha256(extension),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(cuda_home),
        "nvcc": str(nvcc),
        "arch": "sm_86",
        "torch_cuda_arch_list": "8.6",
        "max_jobs": 1,
        "kv_cache_or_flashinfer_bound": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
