#!/usr/bin/env python3
"""Validate a worker benchmark result and print its canonical SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from benchmark_contract import ContractError, canonical_json_bytes, read_json, validate_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    try:
        result = validate_result(read_json(args.result))
    except ContractError as exc:
        print(f"invalid benchmark result: {exc}", file=sys.stderr)
        return 2
    digest = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    print(f"ok sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
