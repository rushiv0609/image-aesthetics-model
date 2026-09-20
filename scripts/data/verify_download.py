#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from image_aesthetics.data.acquisition import verify
def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--verify-only", action="store_true"); a=p.parse_args()
    ok, message = verify(a.data_root/a.dataset)
    print(message); return 0 if ok else 1
if __name__ == "__main__": raise SystemExit(main())
