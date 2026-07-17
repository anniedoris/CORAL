"""Test the drawing2cad grader against seed/solution.py — no full CORAL run.

It instantiates the Grader directly (the way CORAL's daemon would), points it at
the local seed/ (for solution.py) and the local taskdata/ (for the private test
set), and runs it.

Two modes:

  # 1) Inspect what drawing_processor produces on real drawings (no consumer,
  #    no API key, no scoring) — the most useful check while iterating on hints:
  python examples/drawing2cad/scripts/test_grader.py --inspect 3

  # 2) Drive the whole grader end-to-end (runner -> consumer -> scoring):
  python examples/drawing2cad/scripts/test_grader.py --consumer echo --eval-subset 3
  python examples/drawing2cad/scripts/test_grader.py --consumer openai_api --eval-subset 5

Environment: run in an interpreter that has solution.py's libraries (numpy,
opencv, ...) since drawing_processor runs under this same Python. Score mode also
needs `coral` importable (run from the CORAL repo / its venv); the openai_api
consumer additionally needs `openai` + $DRAWING2CAD_CONSUMER_API_KEY. Scoring is
still a placeholder, so score mode returns 0.0 until scoring.py is wired in.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent                 # examples/drawing2cad
REPO_ROOT = EXAMPLE_ROOT.parent.parent     # CORAL repo root
GRADER_SRC = EXAMPLE_ROOT / "grader" / "src"

for _p in (str(REPO_ROOT), str(GRADER_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402  (after sys.path setup)


def load_task_args(task_yaml: Path) -> tuple[dict, int]:
    """Return (grader.args, grader.timeout) from task.yaml."""
    data = yaml.safe_load(Path(task_yaml).read_text()) or {}
    grader = data.get("grader", {}) or {}
    return dict(grader.get("args", {}) or {}), int(grader.get("timeout", 3600) or 3600)


def build_grader(args_dict: dict, timeout: int, codebase: Path, private_dir: Path):
    """Instantiate the Grader with the attributes CORAL's framework would set."""
    from coral.config import GraderConfig
    from drawing2cad_grader.grader import Grader

    config = GraderConfig(
        entrypoint="drawing2cad_grader.grader:Grader",
        timeout=timeout,
        args=args_dict,
    )
    grader = Grader(config)
    grader.codebase_path = str(codebase)   # dir containing solution.py
    grader.private_dir = str(private_dir)  # dir containing taskdata/
    grader.tasks = []
    grader.island_id = None
    return grader


def run_inspect(n: int, codebase: Path, private_dir: Path, args_dict: dict) -> None:
    """Run drawing_processor on the first N samples and print the hints."""
    from drawing2cad_grader import dataset, runner

    testset = private_dir / "taskdata" / "testset"
    samples = dataset.pick_fixed_subset(dataset.load_manifest(testset), n)
    program_file = args_dict.get("program_file", "solution.py")
    timeout = int(args_dict.get("processor_timeout", 60))

    print(f"Inspecting drawing_processor on {len(samples)} sample(s) from {testset}\n")
    scratch = Path(tempfile.mkdtemp(prefix="d2c_inspect_"))
    ok = err = 0
    try:
        for s in samples:
            wd = scratch / s.file_id
            wd.mkdir(parents=True, exist_ok=True)
            try:
                hints = runner.run_processor(
                    str(codebase), str(s.image_path), str(wd), timeout,
                    program_file=program_file,
                )
                ok += 1
                print(f"[{s.file_id}] text={len(hints['text'])} images={len(hints['images'])}")
                for t in hints["text"]:
                    print(f"    text : {t[:140]}")
                for im in hints["images"]:
                    print(f"    image: {im}")
            except runner.RunnerError as e:
                err += 1
                print(f"[{s.file_id}] RUNNER ERROR: {e}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nDone: {ok} ok, {err} error(s).")


def run_score(codebase: Path, private_dir: Path, args_dict: dict, timeout: int) -> None:
    """Drive the full grader and print the resulting ScoreBundle."""
    grader = build_grader(args_dict, timeout, codebase, private_dir)
    print(
        f"Running grader: consumer={args_dict.get('consumer')} "
        f"eval_subset={args_dict.get('eval_subset')}\n"
    )
    bundle = grader.evaluate()
    score = bundle.aggregated
    expl = bundle.scores["eval"].explanation if "eval" in bundle.scores else None
    print("=== RESULT ===")
    print(f"score       : {score}")
    print(f"explanation : {expl}")
    if bundle.feedback:
        print(f"feedback    : {bundle.feedback}")
    if score == 0.0:
        print("\n(note: scoring.py is a placeholder — score stays 0.0 until it is wired in)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task-yaml", default=str(EXAMPLE_ROOT / "task.yaml"))
    ap.add_argument(
        "--codebase", default=str(EXAMPLE_ROOT / "seed"),
        help="Directory containing solution.py.",
    )
    ap.add_argument(
        "--private-dir", default=str(EXAMPLE_ROOT),
        help="Directory containing taskdata/ (grader reads <private-dir>/taskdata/testset).",
    )
    ap.add_argument(
        "--inspect", type=int, metavar="N",
        help="Inspect mode: run drawing_processor on N samples and print hints (no scoring).",
    )
    ap.add_argument(
        "--consumer", choices=["openai_api", "echo", "claude_code"], default="echo",
        help="Override grader.args.consumer (default: echo — no API key needed).",
    )
    ap.add_argument(
        "--eval-subset", type=int, default=10,
        help="Override grader.args.eval_subset (samples scored per run).",
    )
    ap.add_argument(
        "--stream", action="store_true",
        help="claude_code only: live-print the consumer session as it runs.",
    )
    args = ap.parse_args()

    args_dict, timeout = load_task_args(Path(args.task_yaml))
    codebase = Path(args.codebase)
    private_dir = Path(args.private_dir)

    if not (codebase / args_dict.get("program_file", "solution.py")).is_file():
        ap.error(f"solution file not found under {codebase}")
    if not (private_dir / "taskdata" / "testset" / "manifest.jsonl").is_file():
        ap.error(f"test set not found under {private_dir}/taskdata/testset")

    if args.inspect:
        run_inspect(args.inspect, codebase, private_dir, args_dict)

    args_dict["consumer"] = args.consumer
    args_dict["eval_subset"] = args.eval_subset
    if args.stream:
        args_dict["cc_stream"] = True
    run_score(codebase, private_dir, args_dict, timeout)


if __name__ == "__main__":
    main()
