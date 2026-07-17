"""Load the private test set (loose files) and bridge it to the CADBench Evaluator.

On-disk layout, living under `.coral/private/taskdata/testset/`:

    images/<file_id>.png     the engineering drawing (model input)
    stls/<file_id>.stl       the ground-truth mesh (scoring answer)
    manifest.jsonl           {"file_id", "image", "stl", "label"?} per line

The CADBench Evaluator only needs two columns (`file_id`, `stl` bytes); the
drawing images are consumed separately by the consumer, so they are exposed here
as filesystem paths rather than packed into the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    """One test-set item: a drawing (input) paired with its GT mesh (answer)."""

    file_id: str
    image_path: Path  # absolute path to the drawing PNG
    stl_path: Path  # absolute path to the ground-truth STL
    label: str | None = None


def load_manifest(testset_dir: str | Path) -> list[Sample]:
    """Parse manifest.jsonl into Samples with absolute image/stl paths.

    Raises FileNotFoundError if the manifest is missing so the grader can turn
    that into a clean self.fail() rather than an opaque crash.
    """
    testset_dir = Path(testset_dir)
    manifest = testset_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.jsonl not found under {testset_dir}")

    samples: list[Sample] = []
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                Sample(
                    file_id=str(row["file_id"]),
                    image_path=testset_dir / row["image"],
                    stl_path=testset_dir / row["stl"],
                    label=row.get("label"),
                )
            )
    return samples


def pick_fixed_subset(samples: list[Sample], n: int | None) -> list[Sample]:
    """Deterministic subset used as the optimization signal.

    Sorted by file_id and truncated to the first `n`, so every eval scores the
    SAME samples — a stable objective the agents can climb. `n` falsy or >= the
    set size returns all samples (the full-set milestone run).
    """
    ordered = sorted(samples, key=lambda s: s.file_id)
    if not n or n >= len(ordered):
        return ordered
    return ordered[:n]


def build_eval_dataset(samples: list[Sample]):
    """Build the `datasets.Dataset` the CADBench Evaluator consumes.

    The Evaluator indexes `dataset[i]["file_id"]` and `dataset[i]["stl"]` (raw
    STL bytes, loaded via trimesh). Those are the only two columns it reads.
    """
    from datasets import Dataset, Features, Value

    features = Features({"file_id": Value("string"), "stl": Value("binary")})
    return Dataset.from_dict(
        {
            "file_id": [s.file_id for s in samples],
            "stl": [s.stl_path.read_bytes() for s in samples],
        },
        features=features,
    )
