"""Download the image + stl fields from a Drawing_Eval split into the loose
test-set layout the drawing2cad grader expects.

Output layout (under --out):

    <out>/
    ├── images/<file_id>.png     # the engineering drawing (model input)
    ├── stls/<file_id>.stl       # the ground-truth mesh (scoring answer)
    └── manifest.jsonl           # one line per sample: {file_id, image, stl, label?}

The grader reads manifest.jsonl, hands images/<id>.png to drawing_processor +
the consumer, and feeds the stl bytes to the CADBench Evaluator for IoU.

Run in a conda env that has `datasets` and `Pillow` installed, e.g.:

    python examples/drawing2cad/scripts/download_testset.py \
        --split gemcad --image-field image --out examples/drawing2cad/taskdata/testset
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="DeCoDELab/Drawing_Eval", help="HuggingFace dataset id.")
    p.add_argument("--split", default="gemcad", help="Dataset split to pull.")
    p.add_argument(
        "--image-field",
        default="image",
        help="Column holding the drawing image (e.g. 'image' or 'singleview').",
    )
    p.add_argument("--stl-field", default="stl", help="Column holding the ground-truth STL bytes.")
    p.add_argument("--id-field", default="file_id", help="Column holding the unique sample id.")
    p.add_argument(
        "--label-field",
        default="label",
        help="Optional column with a difficulty/category label (skipped if absent).",
    )
    p.add_argument(
        "--out",
        default="examples/drawing2cad/taskdata/testset",
        help="Output directory for the loose test set.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only export the first N samples (default: all).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing --out directory before writing.",
    )
    return p.parse_args()


def safe_id(value: object, idx: int) -> str:
    """Filesystem-safe file_id. Falls back to the row index when the id is empty."""
    text = str(value).strip() if value is not None else ""
    if not text:
        text = f"sample_{idx:05d}"
    # Avoid path separators / whitespace surprises in file names.
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in text)


def save_image(value: object, dest: Path) -> bool:
    """Write a dataset image cell to `dest` as PNG. Returns False if unusable.

    Handles the shapes a datasets column can yield: a decoded PIL.Image, a raw
    {'bytes': ..., 'path': ...} dict, raw bytes, or a filesystem path string.
    """
    if value is None:
        return False

    # Decoded PIL image (the usual case for an Image() feature).
    if hasattr(value, "save"):
        value.save(dest, format="PNG")
        return True

    # Undecoded image dict: {'bytes': ..., 'path': ...}
    if isinstance(value, dict):
        if value.get("bytes"):
            _bytes_to_png(value["bytes"], dest)
            return True
        if value.get("path"):
            value = value["path"]  # fall through to path handling
        else:
            return False

    # Raw encoded bytes.
    if isinstance(value, (bytes, bytearray, memoryview)):
        _bytes_to_png(bytes(value), dest)
        return True

    # Filesystem path string.
    if isinstance(value, str) and Path(value).is_file():
        _bytes_to_png(Path(value).read_bytes(), dest)
        return True

    return False


def _bytes_to_png(raw: bytes, dest: Path) -> None:
    """Re-encode arbitrary image bytes to a normalized PNG via Pillow."""
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        img.load()
        img.save(dest, format="PNG")


def save_stl(value: object, dest: Path) -> bool:
    """Write STL bytes to `dest`. Returns False if the cell has no usable bytes."""
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    elif isinstance(value, str) and Path(value).is_file():
        data = Path(value).read_bytes()
    else:
        return False
    if not data:
        return False
    dest.write_bytes(data)
    return True


def main() -> None:
    args = parse_args()

    out = Path(args.out)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    images_dir = out / "images"
    stls_dir = out / "stls"
    images_dir.mkdir(parents=True, exist_ok=True)
    stls_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} (split={args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    columns = set(ds.column_names)
    for required in (args.image_field, args.stl_field):
        if required not in columns:
            raise SystemExit(
                f"Column {required!r} not in split {args.split!r}. "
                f"Available columns: {sorted(columns)}"
            )
    has_id = args.id_field in columns
    has_label = args.label_field in columns

    total = len(ds) if args.limit is None else min(args.limit, len(ds))
    manifest_path = out / "manifest.jsonl"

    written = 0
    skipped_image = 0
    skipped_stl = 0
    seen_ids: set[str] = set()

    with manifest_path.open("w") as manifest:
        for idx in range(total):
            row = ds[idx]

            fid = safe_id(row.get(args.id_field) if has_id else None, idx)
            if fid in seen_ids:
                fid = f"{fid}_{idx:05d}"  # de-dup collided ids
            seen_ids.add(fid)

            img_dest = images_dir / f"{fid}.png"
            stl_dest = stls_dir / f"{fid}.stl"

            if not save_image(row.get(args.image_field), img_dest):
                skipped_image += 1
                continue
            if not save_stl(row.get(args.stl_field), stl_dest):
                skipped_stl += 1
                img_dest.unlink(missing_ok=True)  # keep the set paired
                continue

            entry = {
                "file_id": fid,
                "image": f"images/{fid}.png",
                "stl": f"stls/{fid}.stl",
            }
            if has_label and row.get(args.label_field) is not None:
                entry["label"] = row[args.label_field]
            manifest.write(json.dumps(entry) + "\n")
            written += 1

            if written % 25 == 0:
                print(f"  ...{written} samples written")

    print(
        f"\nDone. {written} samples -> {out}\n"
        f"  images/  {written} PNGs\n"
        f"  stls/    {written} STLs\n"
        f"  manifest.jsonl ({written} lines)\n"
        f"Skipped: {skipped_image} missing image, {skipped_stl} missing stl."
    )
    if written == 0:
        raise SystemExit(
            "No samples written. Check --image-field/--stl-field names against the "
            "printed 'Available columns' for this split."
        )


if __name__ == "__main__":
    main()
