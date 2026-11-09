"""CLI: python -m cua.run --spec <run_id> --data-root DIR --runs-root DIR [--resume] ...

A run whose COMPLETE.json already exists is reported ("already complete") and exits 0 without
training, so resubmitting a finished job is harmless.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cua.experiments.specs import resolve
from cua.train import train


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cua.run")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deadline-seconds", type=float, default=None)
    ap.add_argument("--max-train-images", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        cfg = resolve(args.spec)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        sys.exit(2)
    complete = args.runs_root / cfg.run_id / "COMPLETE.json"
    if complete.exists():
        print(f"{cfg.run_id}: already complete ({complete}); nothing to do")
        sys.exit(0)
    deadline = None if args.deadline_seconds is None else time.monotonic() + args.deadline_seconds
    code = train(cfg, args.data_root, args.runs_root, resume=args.resume, deadline=deadline,
                 max_train_images=args.max_train_images)
    sys.exit(code)


if __name__ == "__main__":
    main()
