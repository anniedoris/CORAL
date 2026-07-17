"""Download CadQuery ground-truth code from a CADBench-style dataset and export STLs.

For each row:
  1. Write the dataset's `code` field to <code-out-dir>/<file_id>.py.
  2. Append a `cq.exporters.export(...)` call to that file.
  3. Run the file in a subprocess (cwd=<stl-out-dir>) so the STL lands there.
"""

import argparse
import os
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm.auto import tqdm

EXPORT_TEMPLATE = """

import cadquery as cq
cq.exporters.export({variable_name}, '{file_id}.stl')
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="DeCoDELab/TriView2CAD")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--code-field", type=str, default="code")
    parser.add_argument("--id-field", type=str, default="file_id")
    parser.add_argument("--code-out-dir", type=str, default="../data/triview2cad/cadquery")
    parser.add_argument("--stl-out-dir", type=str, default="../data/triview2cad/stls")
    parser.add_argument("--variable-name", type=str, default="result")
    parser.add_argument("--timeout", type=int, default=30, help="Per-script execution timeout, in seconds.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run export even if the target STL already exists.",
    )
    return parser.parse_args()


def export_one(
    file_id: str,
    code: str,
    code_out_dir: str,
    stl_out_dir: str,
    variable_name: str,
    timeout: int,
    overwrite: bool,
):
    code_path = os.path.join(code_out_dir, f"{file_id}.py")
    stl_path = os.path.join(stl_out_dir, f"{file_id}.stl")

    full_code = code.rstrip() + EXPORT_TEMPLATE.format(variable_name=variable_name, file_id=file_id)
    with open(code_path, "w") as f:
        f.write(full_code)

    if not overwrite and os.path.exists(stl_path):
        return file_id, True, "Already exported"

    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(code_path)],
            cwd=stl_out_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return file_id, False, "Timeout"

    if proc.returncode != 0:
        return file_id, False, proc.stderr.strip()[-2000:]

    if not os.path.exists(stl_path):
        return file_id, False, "Script ran but STL was not found"

    return file_id, True, "OK"


def main() -> None:
    args = parse_args()

    os.makedirs(args.code_out_dir, exist_ok=True)
    os.makedirs(args.stl_out_dir, exist_ok=True)

    print(f"Loading dataset {args.dataset} (split={args.split})...", file=sys.stderr)
    dataset = load_dataset(args.dataset, split=args.split)

    rows = [(row[args.id_field], row[args.code_field]) for row in dataset]

    results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                export_one,
                file_id,
                code,
                args.code_out_dir,
                args.stl_out_dir,
                args.variable_name,
                args.timeout,
                args.overwrite,
            ): file_id
            for file_id, code in rows
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Exporting STLs"):
            results.append(future.result())

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"Exported {n_ok}/{len(results)} STLs successfully.", file=sys.stderr)

    failures = [(fid, msg) for fid, ok, msg in results if not ok]
    if failures:
        print(f"{len(failures)} failures:", file=sys.stderr)
        for fid, msg in failures[:20]:
            print(f"  {fid}: {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
