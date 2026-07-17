#!/usr/bin/env python3
"""Evaluate generated CadQuery code (from a JSONL) against local ground-truth STLs."""

import argparse
import os
import sys
import json

# Importing `datasets` before anything that pulls in pymeshlab (via CADBench.Eval)
# avoids an OpenSSL symbol conflict between pymeshlab's bundled libcrypto and
# pyarrow's libssl requirement in some conda environments (e.g. CADBench4).
import datasets  # noqa: F401

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CADBench import pretty_print_metrics
from CADBench.Eval import Evaluator


class LocalSTLDataset:
    """Minimal dataset wrapper so Evaluator can read ground-truth STLs from disk."""

    def __init__(self, gt_dir: str):
        files = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(".stl"))
        if not files:
            raise SystemExit(f"No STL files found in {gt_dir!r}")
        self.file_ids = [os.path.splitext(f)[0] for f in files]
        self.paths = [os.path.join(gt_dir, f) for f in files]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        with open(self.paths[idx], "rb") as f:
            stl_bytes = f.read()
        return {"file_id": self.file_ids[idx], "stl": stl_bytes}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated CadQuery code against local ground-truth STLs using CADBench metrics."
    )
    parser.add_argument("--gt_dir", required=True, help="Directory of ground truth STLs, named <file_id>.stl")
    parser.add_argument("--generated_jsonl", required=True, help="JSONL file with rows of {file_id, generated} CadQuery code")
    parser.add_argument("--output_dir", default=None, help="Directory to write eval_results.json (default: directory of --generated_jsonl)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--variable_name", default="result", help="Python variable name expected to hold the resulting geometry (default: result)")
    parser.add_argument("--timeout", type=int, default=30, help="Per-item code execution timeout, in seconds (default: 30)")
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse JSON on line {line_no} of {path}: {exc}")
            # Normalize to str so file_id always matches the STL-filename-derived
            # ids in LocalSTLDataset, regardless of whether the JSONL stored it
            # as a JSON number or a string.
            row["file_id"] = str(row["file_id"])
            rows.append(row)
    return rows


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.generated_jsonl))
    os.makedirs(output_dir, exist_ok=True)

    outputs = load_jsonl(args.generated_jsonl)
    print(f"Loaded {len(outputs)} generations from {args.generated_jsonl}", file=sys.stderr)

    dataset = LocalSTLDataset(args.gt_dir)
    print(f"Found {len(dataset)} ground-truth STLs in {args.gt_dir}", file=sys.stderr)

    evaluator = Evaluator(
        dataset=dataset,
        num_workers=args.num_workers,
        verbose=True,
        variable_name=args.variable_name,
        timeout=args.timeout,
    )

    metrics, per_item = evaluator.run(outputs)

    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({"per_item": per_item, "summary": metrics}, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    pretty_print_metrics(metrics)


if __name__ == "__main__":
    main()
