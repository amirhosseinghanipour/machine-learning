#!/usr/bin/env python3
"""Reproducibility header: seed every RNG and dump the full environment + git state.

Call `seed_everything(seed)` at the very start of a run, and `environment_report()` to
capture a reproducible run header (log it to your experiment tracker — see
references/experimentation-reproducibility.md). Determinism reproduces ONE run; you still
need MULTIPLE seeds to establish that a finding is real (see references/evaluation-statistics.md).

Usage:
    from repro import seed_everything, environment_report
    seed_everything(0, deterministic=True)        # at run start
    print(json.dumps(environment_report(), indent=2))

Or standalone:  python repro.py [--seed 0] [--deterministic]

Only the stdlib + (optionally) numpy/torch are required; everything degrades gracefully.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict


def seed_everything(seed: int = 0, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch RNGs. Returns the seed for logging.

    deterministic=True makes PyTorch use deterministic algorithms (slower, bit-reproducible).
    Set BEFORE creating data loaders / models. PYTHONHASHSEED only takes effect for the
    *current* process's hashing of new objects; for full hash determinism set it in the
    environment before launching Python.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cuBLAS determinism for matmuls on CUDA >= 10.2
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """Pass as `worker_init_fn` to a PyTorch DataLoader so each worker is seeded.

    Combine with a `torch.Generator` (manual_seed) passed via `generator=` for a fully
    reproducible input pipeline.
    """
    try:
        import numpy as np
        import torch

        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    except ImportError:
        pass


def _git(*args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return out.decode().strip()
    except Exception:
        return None


def environment_report() -> Dict[str, Any]:
    """Return a JSON-serializable snapshot of the environment for the run header."""
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "argv": sys.argv,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            # 'dirty' = uncommitted changes exist; a dirty tree is NOT reproducible
            "dirty": bool(_git("status", "--porcelain")),
        },
        "env": {
            k: os.environ[k]
            for k in ("CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS")
            if k in os.environ
        },
        "packages": {},
    }

    try:
        import numpy as np

        report["packages"]["numpy"] = np.__version__
    except ImportError:
        pass

    try:
        import torch

        report["packages"]["torch"] = torch.__version__
        report["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except ImportError:
        pass

    # Best-effort capture of other common ML packages if installed.
    for name in ("jax", "flax", "sklearn", "transformers", "lightning", "datasets"):
        try:
            mod = __import__(name)
            report["packages"][name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed RNGs and print the environment report.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms.")
    args = parser.parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)
    report = environment_report()
    report["seed"] = args.seed
    report["deterministic"] = args.deterministic
    print(json.dumps(report, indent=2))
