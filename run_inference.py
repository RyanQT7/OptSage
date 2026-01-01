#!/usr/bin/env python3
"""Run OptSage inference on a repository bundle and emit structured results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optsage import OptSage, load_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sample.json"))
    parser.add_argument("--case", help="process one query case_id; default: all")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    repository, queries = load_sample(args.data)
    selected = [item for item in queries if not args.case or item.get("case_id") == args.case]
    if not selected:
        raise SystemExit(f"query case not found: {args.case}")
    system = OptSage(repository, top_k=3, tau=0.70, epsilon=0.70)
    results = [system.diagnose(item) for item in selected]
    rendered = json.dumps({
        "knowledge_version": repository.version,
        "results": results,
    }, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
