"""Local dev harness: run drawing_processor on the public dev drawings and show
its hints — no grader, no consumer, no API, no cost.

This is the fast loop for developing drawing_processor: run it on real drawings,
read the text hints, and open the 2D image hints it wrote. Only spend `coral eval`
once you want to measure whether your hints actually improve the consumer's IoU.

    python dev_inspect.py                 # all dev drawings
    python dev_inspect.py --limit 3       # first 3
    python dev_inspect.py --id 00033822   # one drawing by file_id
    python dev_inspect.py --out out/      # where 2D image hints are written

Image hints are written under --out (default: inspect_out/) and kept, so you can
open them. The matching ground-truth STL for each drawing is in dev/stls/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from solution import drawing_processor  # noqa: E402  (after sys.path setup)

DEV = HERE / "dev"


def load_dev() -> list[dict]:
    """Read dev/manifest.jsonl, or fall back to globbing dev/images/*.png."""
    manifest = DEV / "manifest.jsonl"
    if manifest.is_file():
        return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    return [
        {"file_id": p.stem, "image": f"images/{p.name}"}
        for p in sorted((DEV / "images").glob("*.png"))
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, default=None, help="only the first N dev drawings")
    ap.add_argument("--id", dest="file_id", default=None, help="run one drawing by file_id")
    ap.add_argument(
        "--out", default=str(HERE / "inspect_out"),
        help="dir for the 2D image hints (kept, not cleaned up)",
    )
    args = ap.parse_args()

    rows = load_dev()
    if not rows:
        sys.exit("No dev drawings found under dev/. Is seed/dev/ populated?")
    if args.file_id:
        rows = [r for r in rows if r["file_id"] == args.file_id]
        if not rows:
            sys.exit(f"No dev sample with file_id {args.file_id!r}")
    if args.limit:
        rows = rows[: args.limit]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for r in rows:
        drawing = DEV / r["image"]
        workdir = out_root / r["file_id"]
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            hints = drawing_processor(str(drawing), str(workdir))
        except Exception as e:  # noqa: BLE001 — surface any processor error, keep going
            print(f"[{r['file_id']}] ERROR: {e}\n")
            continue

        text = list(getattr(hints, "text", []) or [])
        images = list(getattr(hints, "images", []) or [])
        print(f"[{r['file_id']}] {drawing.name}")
        print(f"  text hints ({len(text)}):")
        for t in text:
            print(f"    - {t}")
        print(f"  image hints ({len(images)}):")
        for im in images:
            print(f"    - {im}")
        print()

    print(f"2D image hints written under: {out_root}")


if __name__ == "__main__":
    main()
