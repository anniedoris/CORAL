from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================================
# DO NOT EDIT — frozen contract.
# The grader depends on this class name and its two field names (`text`,
# `images`). Changing, renaming, removing, or adding fields will break scoring.
# ============================================================================
@dataclass
class DrawingHints:
    """Hints passed to a CAD-writing consumer, like Claude Code or another VLM that is going from engineering drawing to CadQuery code.

    An empty DrawingHints() == the baseline (consumer works from the raw
    drawing alone). The grader reads exactly these two fields:

      text   — 1D: strings injected directly into the consumer's prompt.
      images — 2D: paths to image files written under `workdir`, attached to
               the consumer as image inputs. Paths must live inside `workdir`.
    """

    text: list[str] = field(default_factory=list)      # 1D: prompt text
    images: list[str] = field(default_factory=list)    # 2D: image paths under workdir


# ============================================================================
# EDIT HERE — this is the function CORAL agents evolve.
# Keep the signature `(drawing_path, workdir) -> DrawingHints` fixed; fill the
# body (and add any helper functions/imports you need). Return your hints packed
# into DrawingHints.text and DrawingHints.images.
# ============================================================================
import os


def _content_crop_bbox(gray):
    """Bounding box (x0,y0,x1,y1) of the drawing views/dimensions in the sheet.

    Removes the outer border frame (fixed margin) and the title block (the ink
    blob anchored at the interior's bottom-right corner), then returns the bbox
    of everything else — the orthographic + isometric views and their dimensions.
    """
    import cv2
    import numpy as np

    h, w = gray.shape
    m = 0.035  # drop the border frame + corner arrows
    x0, x1 = int(w * m), int(w * (1 - m))
    y0, y1 = int(h * m), int(h * (1 - m))
    interior = gray[y0:y1, x0:x1]
    Hh, Ww = interior.shape

    ink = (interior < 128).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(closed, 8)

    # Title block = the largest component whose bbox reaches the bottom-right corner.
    tb = None
    best = 0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if (x + ww) > Ww * 0.985 and (y + hh) > Hh * 0.985 and area > best:
            best = area
            tb = i

    boxes = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 80 or i == tb:
            continue
        boxes.append((x, y, x + ww, y + hh))
    if not boxes:
        return None

    bx0 = min(b[0] for b in boxes)
    by0 = min(b[1] for b in boxes)
    bx1 = max(b[2] for b in boxes)
    by1 = max(b[3] for b in boxes)
    return (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)


def _view_clusters(gray, content_bbox):
    """Bounding boxes (full-image coords) of the individual view clusters.

    A "view cluster" is one orthographic/section/isometric view plus its
    surrounding dimension annotations, merged into a blob by a large
    morphological close. Returns clusters sorted by area (largest first),
    dropping tiny fragments and title-block-like blobs.
    """
    import cv2
    import numpy as np

    cx0, cy0, cx1, cy1 = content_bbox
    region = gray[cy0:cy1, cx0:cx1]
    Hh, Ww = region.shape
    if Hh < 10 or Ww < 10:
        return []

    ink = (region < 128).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(closed, 8)

    area_min = 0.004 * Hh * Ww
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < area_min:
            continue
        # Drop a title-block-like blob: anchored to the bottom-right corner AND
        # covering a large, roughly rectangular fraction (a filled table).
        fill = area / float(max(1, w * h))
        br_corner = (x + w) > Ww * 0.97 and (y + h) > Hh * 0.97
        if br_corner and fill > 0.55 and w > 0.25 * Ww:
            continue
        boxes.append((cx0 + x, cy0 + y, cx0 + x + w, cy0 + y + h, area))
    boxes.sort(key=lambda b: -b[4])
    return [(b[0], b[1], b[2], b[3]) for b in boxes]


def _upscale(crop, target):
    """Upscale a grayscale crop so its long edge is ~target px (cap 3x)."""
    import cv2

    ch, cw = crop.shape
    scale = min(3.0, target / max(ch, cw))
    if scale > 1.0:
        crop = cv2.resize(
            crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC
        )
    return crop


def _ortho_only_bbox(gray, content_bbox):
    """Content bbox with the isometric view removed (higher-res ortho views).

    The isometric is the rightmost view cluster separated by a clear horizontal
    gap. Dropping it from OUR crop lets the dimensioned orthographic views fill
    the frame (~2x resolution); the isometric is still visible in the original
    drawing the consumer also receives. Returns None (→ use the full crop) when
    there are < 3 clusters or no clearly-separated rightmost cluster, so we never
    accidentally amputate a real ortho view.
    """
    clusters = _view_clusters(gray, content_bbox)
    if len(clusters) < 3:
        return None
    iso = max(clusters, key=lambda c: c[2])  # rightmost by right edge
    others = [c for c in clusters if c is not iso]
    others_r = max(c[2] for c in others)
    content_w = content_bbox[2] - content_bbox[0]
    if (iso[0] - others_r) < 0.06 * content_w:  # not clearly separated
        return None
    ox0 = min(c[0] for c in others)
    oy0 = min(c[1] for c in others)
    oy1 = max(c[3] for c in others)
    return (ox0, oy0, others_r, oy1)


_OCR_READER = None


def _get_ocr_reader():
    """Lazily build a cached easyocr Reader (English, CPU). None if unavailable."""
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr

        _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER


def _extract_dimensions(crop_gray):
    """OCR a content crop and return a normalized list of dimension labels.

    Parts here are all sub-2-inch, so a bare integer read is almost always a
    decimal whose leading dot easyocr dropped (".44" -> "44"); we restore it.
    Ø (diameter) and R (radius) prefixes are kept when detected. Best-effort:
    returns [] on any failure so the caller still ships the crop.
    """
    import re

    try:
        import cv2

        reader = _get_ocr_reader()
        if reader is None:
            return []
        up = crop_gray
        h, w = up.shape
        if max(h, w) < 1600:
            s = 1800.0 / max(h, w)
            up = cv2.resize(up, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        tokens = reader.readtext(up, detail=0, paragraph=False)
    except Exception:
        return []

    out = []
    seen = set()
    for tok in tokens:
        s = str(tok).strip().replace(" ", "")
        s = s.replace("⌀", "Ø").replace("∅", "Ø")
        # optional R/Ø prefix, then digits with optional single decimal point
        m = re.match(r"^([RrØ]?)Ø?\.?([0-9]+(?:\.[0-9]+)?)$", s)
        if not m:
            continue
        pre = "R" if m.group(1) in ("R", "r") else ("Ø" if "Ø" in s[:2] else "")
        num = m.group(2)
        # Restore a dropped leading decimal: parts are sub-2-inch, so a bare
        # integer read (".44" -> "44") is a decimal missing its dot. But a 3+
        # digit bare read (e.g. "225") is an unreliable misread — drop it.
        if "." not in num and float(num) >= 1:
            if len(num) >= 3:
                continue
            num = "." + num
        try:
            val = float(num)
        except ValueError:
            continue
        if not (0 < val <= 2.5):  # drop implausible reads
            continue
        label = pre + num
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def drawing_processor(drawing_path: str, workdir: str) -> DrawingHints:
    """Turn an engineering-drawing image into hints for the CAD-writing consumer.

    FINAL config = the proven v1 overview crop: a single whitespace-trimmed,
    upscaled legibility image (border frame + title block removed) with a one-line
    caption. This is the only reliably score-positive hint (baseline 0.4579 → crop
    up to 0.5792). Extensively tested alternatives — per-view tiles (v2), verbose
    reading/validity text (v3/v4), and an OCR dimension checklist (v5) — did NOT beat
    it: the consumer is very noisy (temperature ~1.0; identical code spans 0.41–0.58)
    and the residual error is the model's 3D-reasoning on complex parts, which no
    perception hint fixes. Kept minimal on purpose. See notes/synthesis-what-works.md.
    (_extract_dimensions and the OCR helpers are retained above as reference only.)
    """
    hints = DrawingHints()
    try:
        import cv2

        gray = cv2.imread(drawing_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return hints
        bbox = _content_crop_bbox(gray)
        if bbox is None:
            return hints
        # Structural attempt: crop to the dimensioned orthographic views only
        # (drop the isometric), for higher effective resolution. Falls back to the
        # full content bbox when the isometric can't be confidently isolated.
        try:
            ortho = _ortho_only_bbox(gray, bbox)
        except Exception:
            ortho = None
        crop_bbox = ortho if ortho is not None else bbox
        H, W = gray.shape
        pad = 18
        ox0 = max(0, crop_bbox[0] - pad)
        oy0 = max(0, crop_bbox[1] - pad)
        ox1 = min(W, crop_bbox[2] + pad)
        oy1 = min(H, crop_bbox[3] + pad)
        overview = _upscale(gray[oy0:oy1, ox0:ox1], 1600)
        out = os.path.join(workdir, "overview.png")
        cv2.imwrite(out, overview)
        hints.images.append(out)

        note = (
            "The attached image is a zoomed-in, whitespace-trimmed crop of the "
            "engineering drawing's dimensioned views (border frame and title block "
            "removed) at higher effective resolution — use it to read the geometry "
            "and dimension values accurately."
        )
        if ortho is not None:
            note += (
                " It focuses on the orthographic views; the isometric/3D view of the "
                "same part is in the full drawing above."
            )
        hints.text.append(note)
    except Exception:
        return DrawingHints()
    return hints
