"""ctypes bridge for the two Mojo slicing kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_ANNDATA_LIB") or os.path.join(ROOT, "dist", "libmojo-anndata.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mad_dense_take_f64": ([I, I, I, I, I, I, I], None),
    "mad_dense_take_f64_parallel": ([I, I, I, I, I, I, I], None),
    "mad_csr_take_f64": ([I, I, I, I, I, I, I, I, I], I),
    "mad_csr_take_f64_i32": ([I, I, I, I, I, I, I, I, I], I),
}
_loaded: ctypes.CDLL | None = None


def build() -> str:
    if os.environ.get("MOJO_ANNDATA_LIB") and os.path.exists(LIB):
        return LIB
    sources = [os.path.join(ROOT, "src", "capi.mojo")]
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
        return LIB
    proc = subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], cwd=ROOT,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
    return _loaded


def addr(array) -> int:
    return int(array.ctypes.data)
