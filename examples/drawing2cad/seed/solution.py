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
def drawing_processor(drawing_path: str, workdir: str) -> DrawingHints:
    """Turn an engineering-drawing image into hints for the CAD-writing consumer.

    BASELINE: returns no hints. The consumer sees only the raw drawing, so the
    seed score is the "no help" reference.
    """
    return DrawingHints()
