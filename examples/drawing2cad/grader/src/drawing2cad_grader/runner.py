"""Run the agent's drawing_processor in an isolated subprocess.

drawing_processor is untrusted, agent-authored code executed every eval, so we
run it with a separate interpreter invocation with a hard timeout, 
communicating results as a single JSON line on stdout.
A hang or crash fails one sample instead of taking down the grader.

The processor writes derived images into `workdir` and returns their paths; we
validate those paths live inside `workdir` before handing them to the consumer,
so an agent cannot smuggle in arbitrary filesystem paths.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Marker so we can find our JSON payload even if the agent's code prints noise.
_SENTINEL = "__DRAWING2CAD_HINTS__"
_PROCESSOR_FN = "drawing_processor"


class RunnerError(RuntimeError):
    """drawing_processor crashed, timed out, or produced unusable output."""


def _build_script(codebase_path: str, module: str, drawing_path: str, workdir: str) -> str:
    """Python source that imports the agent module and calls drawing_processor."""
    return (
        "import json, sys\n"
        f"sys.path.insert(0, {codebase_path!r})\n"
        f"import {module} as _m\n"
        f"_hints = _m.{_PROCESSOR_FN}({drawing_path!r}, {workdir!r})\n"
        "_text = list(getattr(_hints, 'text', []) or [])\n"
        "_images = list(getattr(_hints, 'images', []) or [])\n"
        f"print({_SENTINEL!r} + json.dumps("
        "{'text': [str(t) for t in _text], 'images': [str(p) for p in _images]}))\n"
    )


def _parse_payload(stdout: str) -> dict:
    """Extract the JSON hints payload from stdout (last sentinel line wins)."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(_SENTINEL):
            return json.loads(line[len(_SENTINEL):])
    raise RunnerError("drawing_processor produced no hints payload on stdout")


def _valid_images(image_paths: list[str], workdir: str) -> list[str]:
    """Keep only images that exist and resolve inside `workdir`."""
    root = Path(workdir).resolve()
    kept: list[str] = []
    for p in image_paths:
        try:
            resolved = Path(p).resolve()
            resolved.relative_to(root)  # raises if outside workdir
        except (ValueError, OSError):
            continue
        if resolved.is_file():
            kept.append(str(resolved))
    return kept


def run_processor(
    codebase_path: str,
    drawing_path: str,
    workdir: str,
    timeout: int,
    *,
    program_file: str = "solution.py",
    python_cmd: list[str] | None = None,
) -> dict:
    """Invoke drawing_processor(drawing_path, workdir); return {"text", "images"}.

    Runs in the grader interpreter by default (python_cmd=None), so the agent may
    only use libraries pre-installed in the grader venv. Raises RunnerError on
    crash/timeout/malformed output; the caller decides whether to fail the sample
    or fall back to empty hints.
    """
    module = Path(program_file).stem
    script = _build_script(codebase_path, module, drawing_path, workdir)
    cmd = [*(python_cmd or [sys.executable]), "-c", script]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RunnerError(f"drawing_processor timed out after {timeout}s")

    if proc.returncode != 0:
        raise RunnerError(f"drawing_processor crashed: {proc.stderr.strip()[-800:]}")

    payload = _parse_payload(proc.stdout)
    return {
        "text": [str(t) for t in payload.get("text", [])],
        "images": _valid_images([str(p) for p in payload.get("images", [])], workdir),
    }
