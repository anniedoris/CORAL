"""Score generated CadQuery against ground-truth STLs via the vendored CADBench.

Scoring runs the CADBench Evaluator in a SEPARATE subprocess (this same file, run
as a script, reading a JSON payload on stdin). The reason: the native mesh libs
CADBench pulls in (pymeshlab / open3d / OCP) corrupt the heap on interpreter
teardown — `malloc_consolidate(): unaligned fastbin chunk detected`, SIGABRT. If
they load into the main grader worker, that worker aborts on exit *after* computing
a valid score, and CORAL discards the result as a crash. Isolating them here means
the main worker never imports them and exits cleanly; the subprocess prints its
metrics and then os._exit(0)s to skip the crashing teardown entirely.

CADBench lives (vendored) next to this package under CADBenchEval/; the worker adds
it to sys.path so `import CADBench` resolves without installing it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Vendored copy: .../drawing2cad_grader/CADBenchEval (contains the CADBench pkg).
VENDORED_CADBENCH = Path(__file__).resolve().parent / "CADBenchEval"
_THIS_FILE = Path(__file__).resolve()
_METRICS_SENTINEL = "__DRAWING2CAD_METRICS__"


@dataclass
class ScoreBreakdown:
    """Aggregate result the grader turns into self.score(value, explanation)."""

    score: float  # CORAL objective: adjusted-median Aligned IoU (invalid -> 0)
    n: int  # number of samples scored
    vsr: float = 0.0  # valid-solid rate, %
    mean_iou_success: float = 0.0  # mean Aligned IoU over valid solids only
    note: str | None = None  # surfaced in the leaderboard explanation


class _LocalSTLDataset:
    """Picklable dataset the Evaluator reads GT STL bytes from, keyed by file_id.

    Only picklable attributes (str lists), because Evaluator pickles it out to its
    ProcessPool workers. Same shape as LocalSTLDataset in scripts/eval_local.py:
    __len__ + __getitem__ -> {file_id, stl}.
    """

    def __init__(self, file_ids: list[str], stl_paths: list[str]) -> None:
        self.file_ids = list(file_ids)
        self.stl_paths = [str(p) for p in stl_paths]

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> dict:
        with open(self.stl_paths[idx], "rb") as f:
            return {"file_id": self.file_ids[idx], "stl": f.read()}


def score_outputs(
    samples,
    outputs: list[dict],
    *,
    cadbench_path: str | None = None,
    variable_name: str = "result",
    num_workers: int = 8,
    timeout: int = 30,
) -> ScoreBreakdown:
    """Score generated CadQuery against the samples' ground-truth STLs.

    Runs the CADBench Evaluator in a subprocess (see module docstring) so the
    native mesh libs never load into the main grader worker.

    Args:
        samples: list[dataset.Sample] for the scored subset (uses file_id + stl_path).
        outputs: list of {"file_id", "generated": <cadquery code str>}.
        cadbench_path: override for the CADBenchEval dir (default: vendored copy).
        variable_name: the CadQuery result variable the code must define.
        num_workers / timeout: passed to the CADBench Evaluator.
    """
    payload = {
        "cadbench_path": str(cadbench_path) if cadbench_path else str(VENDORED_CADBENCH),
        "samples": [{"file_id": s.file_id, "stl": str(s.stl_path)} for s in samples],
        "outputs": outputs,
        "variable_name": variable_name,
        "num_workers": num_workers,
        "timeout": timeout,
    }
    # Generous wall-clock ceiling for the whole scoring pass (the Evaluator also
    # enforces per-item `timeout` internally). Scoring is parallel, so this is far
    # more than needed; it just prevents an indefinite hang.
    overall_timeout = max(600, (len(outputs) // max(1, num_workers) + 1) * timeout * 3 + 120)

    try:
        proc = subprocess.run(
            [sys.executable, str(_THIS_FILE)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=overall_timeout,
        )
    except subprocess.TimeoutExpired:
        return ScoreBreakdown(0.0, len(outputs), note=f"scoring timed out after {overall_timeout}s")

    metrics = _parse_metrics(proc.stdout)
    if metrics is None:
        return ScoreBreakdown(
            0.0,
            len(outputs),
            note=(
                f"scoring produced no metrics (exit {proc.returncode}); "
                f"stderr: {proc.stderr.strip()[-300:]}"
            ),
        )
    return ScoreBreakdown(
        score=float(metrics["Adjusted Median"]["Aligned IoU"]),
        n=len(outputs),
        vsr=float(metrics["VSR"]),
        mean_iou_success=float(metrics["Mean"]["Aligned IoU"]),
    )


def _parse_metrics(stdout: str) -> dict | None:
    """Pull the sentinel-tagged metrics JSON out of the worker's stdout."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(_METRICS_SENTINEL):
            try:
                return json.loads(line[len(_METRICS_SENTINEL):])
            except json.JSONDecodeError:
                return None
    return None


def _run_scoring_worker() -> None:
    """Subprocess entrypoint: read payload on stdin, run Evaluator, print metrics.

    Ends with os._exit(0) to skip interpreter teardown — the native mesh libs
    corrupt the heap on exit, which would otherwise abort this process (SIGABRT).
    """
    payload = json.load(sys.stdin)

    root = payload["cadbench_path"]
    if root not in sys.path:
        sys.path.insert(0, root)

    # Import datasets before pymeshlab (pulled in by CADBench.Eval) to dodge an
    # OpenSSL symbol clash in some conda envs — see scripts/eval_local.py.
    try:
        import datasets  # noqa: F401
    except ImportError:
        pass
    from CADBench.Eval import Evaluator

    dataset = _LocalSTLDataset(
        [s["file_id"] for s in payload["samples"]],
        [s["stl"] for s in payload["samples"]],
    )
    metrics, _per_item = Evaluator(
        dataset=dataset,
        num_workers=payload["num_workers"],
        verbose=False,
        variable_name=payload["variable_name"],
        timeout=payload["timeout"],
    ).run(payload["outputs"])

    sys.stdout.write(_METRICS_SENTINEL + json.dumps(metrics) + "\n")
    sys.stdout.flush()
    os._exit(0)  # skip teardown -> avoid the native heap-corruption abort


if __name__ == "__main__":
    _run_scoring_worker()
