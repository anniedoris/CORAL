"""The CAD-writing consumer: (drawing + hints) -> CadQuery code.

This is the frozen "coding agent" whose output the hints are meant to improve.
It sits behind the `Consumer` protocol so the implementation is swappable via
`grader.args.consumer` without touching the grader:

    openai_api  -> OpenAIConsumer     (Claude Opus via an OpenAI-compatible API)
    echo        -> EchoConsumer       (no-API stub, for dry runs / plumbing tests)
    claude_code -> ClaudeCodeConsumer (agentic Claude Code via `claude -p`)

The prompt is held fixed here so the ONLY thing that varies between evals is the
agent's hints. The API key comes from an environment variable, never from
task.yaml (which is agent-visible).

generate() returns a GenerationResult (code + raw output + a compact prompt
record) so the grader can log per-sample consumer I/O to generations.jsonl.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

# Fixed prompts given to the consumer. Hints are injected into the user prompt.
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert CAD engineer and 3D modeling specialist with deep "
    "knowledge of mechanical design, geometric modeling, and CadQuery."
)
DEFAULT_USER_PROMPT = """Generate CadQuery Python code to create this 3D CAD model
based on the provided image.

Requirements:
- Use CadQuery syntax
- Create a variable called 'result' containing the final geometry
- Include all necessary imports
- Use parametric dimensions where appropriate
- The code must be executable and create valid solid geometry
- The last line must be: result = <your_final_geometry>
- Do NOT include show_object, exportStep, or any display/export calls
- ALWAYS use method chaining for Workplane operations -- never call methods like moveTo, lineTo, close, extrude, box, circle etc. without capturing the return value. Every Workplane method returns a new object; chain them or reassign, like:
  result = (cq.Workplane()
      .moveTo(0, 0)
      .lineTo(10, 0)
      .close()
      .extrude(5))

Return ONLY the Python code, no explanations."""


@dataclass
class GenerationResult:
    """One consumer call: the extracted code plus what produced it (for logging).

    `prompt` is a compact, human-readable record of the request — NOT the raw API
    payload, which embeds megabytes of base64 image data.
    """

    code: str  # processed CadQuery — what the Evaluator scores
    raw_response: str  # the model's full raw text output, before code extraction
    prompt: dict  # {system, user_text, images, model}


@runtime_checkable
class Consumer(Protocol):
    """Turns a drawing plus the agent's hints into CadQuery code."""

    def generate(
        self,
        drawing_path: str,
        hints_text: list[str],
        hint_image_paths: list[str],
    ) -> GenerationResult:
        ...


def _encode_image(path: str) -> str:
    """Base64 data-URL payload for an image file (PNG)."""
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")


def _extract_code(raw: str) -> str:
    """Pull Python out of a fenced code block, falling back to the raw text.

    Mirrors APIInferenceEngine.extract_code in the drawing_eval repo.
    """
    for pattern in (r"```python\s*\n(.*?)```", r"```\w+\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1).strip()
    return raw.strip()


def _prompt_record(system: str, user_text: str, drawing_path: str, hint_image_paths: list[str], model: str) -> dict:
    """Compact, base64-free record of the request for logging."""
    return {
        "system": system,
        "user_text": user_text,
        "images": [f"drawing:{drawing_path}", *[f"hint:{p}" for p in hint_image_paths]],
        "model": model,
    }


class OpenAIConsumer:
    """Claude Opus (or any OpenAI-compatible chat model) as the fixed consumer."""

    def __init__(
        self,
        model: str,
        api_endpoint: str,
        api_key: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt: str = DEFAULT_USER_PROMPT,
        sampling: dict | None = None,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=api_endpoint)
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.sampling = sampling or {}

    def generate(
        self,
        drawing_path: str,
        hints_text: list[str],
        hint_image_paths: list[str],
    ) -> GenerationResult:
        # Order: the drawing first, then any hint images, then the text block.
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": _encode_image(drawing_path)}}
        ]
        for img in hint_image_paths:
            content.append({"type": "image_url", "image_url": {"url": _encode_image(img)}})

        user_text = self.user_prompt
        if hints_text:
            user_text += "\n\nAdditional analysis of the drawing:\n" + "\n".join(hints_text)
        content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **self.sampling
        )
        raw = response.choices[0].message.content or ""
        return GenerationResult(
            code=_extract_code(raw),
            raw_response=raw,
            prompt=_prompt_record(
                self.system_prompt, user_text, drawing_path, hint_image_paths, self.model
            ),
        )


class EchoConsumer:
    """No-API stub: returns a trivial valid CadQuery solid.

    Lets you exercise the full grader loop (manifest -> processor -> scoring)
    without a model or an API key. Every sample yields the same unit box.
    """

    def generate(
        self,
        drawing_path: str,
        hints_text: list[str],
        hint_image_paths: list[str],
    ) -> GenerationResult:
        code = "import cadquery as cq\nresult = cq.Workplane('XY').box(1, 1, 1)\n"
        user_text = "(echo consumer — no model call)"
        if hints_text:
            user_text += "\n\nAdditional analysis of the drawing:\n" + "\n".join(hints_text)
        return GenerationResult(
            code=code,
            raw_response=code,
            prompt=_prompt_record(
                "(echo)", user_text, drawing_path, hint_image_paths, "echo"
            ),
        )


CLAUDE_CODE_INSTRUCTIONS = """An engineering drawing is saved as `drawing.png` in your working directory.{hint_images}{hint_notes}

Read the drawing (and any hint images), then write CadQuery Python code that reconstructs the 3D part as accurately as possible.

Write your final code to a file named `output.py` in this directory. It must:
- `import cadquery as cq`
- define a variable named `{variable_name}` holding the final solid
- be runnable and produce a valid solid
- NOT call show_object, exportStep, or any display/export calls

Verify it executes and produces a valid solid by running `{python_bin} output.py` and fix any errors before finishing (use exactly that interpreter — it has cadquery installed). When done, make sure `output.py` contains your best CadQuery."""


def _cc_result_text(stdout: str) -> str:
    """Best-effort final assistant text from `claude -p` (json or stream-json)."""
    # Captured mode: a single JSON object.
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data.get("result") or ""
    except (json.JSONDecodeError, TypeError):
        pass
    # Stream mode: find the trailing result event among the NDJSON lines.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "result":
            return ev.get("result") or ""
    return stdout


def _summarize_tool_result(content) -> str:
    """One-line summary of a tool_result — images become `(image)`, not base64."""
    if isinstance(content, str):
        return content.strip()[:200] or "(empty)"
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item)[:100])
            elif item.get("type") == "image":
                parts.append("(image)")
            elif item.get("type") == "text":
                parts.append(item.get("text", "").strip()[:200])
            else:
                parts.append(item.get("type", "?"))
        return " ".join(p for p in parts if p)[:200] or "(empty)"
    return str(content)[:200]


def _print_cc_event(line: str) -> None:
    """Print a readable one-liner for a `claude -p --output-format stream-json` event."""
    line = line.strip()
    if not line:
        return
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return
    kind = ev.get("type")
    if kind == "system":
        # Announce init once; ignore the stream of thinking-token progress events.
        if ev.get("subtype") == "init":
            print("[claude] session started", flush=True)
    elif kind == "assistant":
        for block in ev.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "text" and block.get("text", "").strip():
                print(f"[claude] {block['text'].strip()}", flush=True)
            elif btype == "thinking":
                print("[claude] (thinking…)", flush=True)
            elif btype == "tool_use":
                inp = json.dumps(block.get("input", {}))
                print(f"[claude] -> {block.get('name')}: {inp[:120]}", flush=True)
    elif kind == "user":
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                print(f"[claude]    <- {_summarize_tool_result(block.get('content'))}", flush=True)
    elif kind == "result":
        print(
            f"[claude] done: {ev.get('subtype', '')} "
            f"(turns={ev.get('num_turns')}, cost=${ev.get('total_cost_usd', '?')})",
            flush=True,
        )


class ClaudeCodeConsumer:
    """Agentic Claude Code (`claude -p`) as the fixed consumer.

    Stages the drawing + hint images as files in a scratch dir, runs a headless
    `claude -p` session that reads them and writes CadQuery to `output.py`, then
    reads that file back (falling back to the final message if it wasn't written).

    Auth is the `claude` CLI's own — a subscription unless ANTHROPIC_API_KEY is
    set in the grader environment — NOT the openai_api key. Its edge over the API
    consumer is self-repair: it can run its own CadQuery, see errors, and fix them.
    """

    def __init__(
        self,
        model: str = "opus",
        *,
        max_turns: int = 30,
        timeout: int = 300,
        variable_name: str = "result",
        python_bin: str | None = None,
        stream: bool = False,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.timeout = timeout
        self.variable_name = variable_name
        # Interpreter Claude should self-test with. Defaults to the grader's own
        # python (sys.executable) — the same one the Evaluator scores with, so it
        # is guaranteed to have cadquery, regardless of what `python` is on PATH.
        self.python_bin = python_bin or sys.executable
        # When True, stream the session's steps to the terminal live.
        self.stream = stream

    def _build_prompt(self, hint_image_names: list[str], hints_text: list[str]) -> str:
        hint_images = ""
        if hint_image_names:
            hint_images = "\nAdditional analysis images are available: " + ", ".join(
                f"`{n}`" for n in hint_image_names
            ) + "."
        hint_notes = ""
        if hints_text:
            hint_notes = "\n\nNotes from a perception tool:\n" + "\n".join(hints_text)
        return CLAUDE_CODE_INSTRUCTIONS.format(
            hint_images=hint_images,
            hint_notes=hint_notes,
            variable_name=self.variable_name,
            python_bin=self.python_bin,
        )

    def _base_cmd(self, prompt: str, output_format: str, extra: list[str]) -> list[str]:
        return [
            "claude", "-p", prompt,
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--dangerously-skip-permissions",  # headless: no human to approve tools
            "--output-format", output_format,
            *extra,
        ]

    def _run_captured(self, prompt: str, workdir: Path) -> str:
        """Run to completion, capturing the single JSON result (no live output)."""
        cmd = self._base_cmd(prompt, "json", [])
        try:
            proc = subprocess.run(
                cmd, cwd=str(workdir), capture_output=True, text=True, timeout=self.timeout
            )
            return proc.stdout
        except subprocess.TimeoutExpired:
            return f"(claude -p timed out after {self.timeout}s)"

    def _run_stream(self, prompt: str, workdir: Path) -> str:
        """Stream the session live (one event per line), printing each readably."""
        cmd = self._base_cmd(prompt, "stream-json", ["--verbose"])
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd, cwd=str(workdir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        killer = threading.Timer(self.timeout, proc.kill)  # timeout guard for the stream
        killer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line)
                _print_cc_event(line)
            proc.wait()
        finally:
            killer.cancel()
        return "".join(lines)

    def generate(
        self,
        drawing_path: str,
        hints_text: list[str],
        hint_image_paths: list[str],
    ) -> GenerationResult:
        workdir = Path(tempfile.mkdtemp(prefix="cc_consumer_"))
        try:
            shutil.copy(drawing_path, workdir / "drawing.png")
            hint_names: list[str] = []
            for i, h in enumerate(hint_image_paths):
                name = f"hint_{i}{Path(h).suffix or '.png'}"
                shutil.copy(h, workdir / name)
                hint_names.append(name)

            prompt = self._build_prompt(hint_names, hints_text)
            raw = (
                self._run_stream(prompt, workdir)
                if self.stream
                else self._run_captured(prompt, workdir)
            )

            out_file = workdir / "output.py"
            if out_file.is_file():
                code = out_file.read_text()
            else:
                # Claude didn't write the file — salvage code from its final message.
                code = _extract_code(_cc_result_text(raw))

            return GenerationResult(
                code=code,
                raw_response=raw,
                prompt=_prompt_record(
                    "(claude_code)", prompt, drawing_path, hint_image_paths, self.model
                ),
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def make_consumer(args) -> Consumer:
    """Build the consumer selected by grader.args.consumer."""
    kind = args.get("consumer", "openai_api")

    if kind == "echo":
        return EchoConsumer()
    if kind == "claude_code":
        if shutil.which("claude") is None:
            raise RuntimeError(
                "claude_code consumer needs the `claude` CLI on PATH "
                "(install Claude Code + `claude login`, or set ANTHROPIC_API_KEY)"
            )
        return ClaudeCodeConsumer(
            model=args.get("cc_model", "opus"),  # CLI alias, not the API model id
            max_turns=int(args.get("cc_max_turns", 30)),
            timeout=int(args.get("cc_timeout", 300)),
            variable_name=args.get("variable_name", "result"),
            python_bin=args.get("cc_python"),  # None -> the grader's own python
            stream=bool(args.get("cc_stream", False)),  # live-print the session
        )
    if kind == "openai_api":
        key_env = args.get("api_key_env", "DRAWING2CAD_CONSUMER_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"openai_api consumer needs an API key in ${key_env} "
                "(set it in the grader environment, not in task.yaml)"
            )
        sampling: dict = {}
        if args.get("temperature") is not None:
            sampling["temperature"] = float(args["temperature"])
        if args.get("max_tokens") is not None:
            sampling["max_tokens"] = int(args["max_tokens"])
        return OpenAIConsumer(
            model=args["model"],
            api_endpoint=args.get("api_endpoint", "https://api.anthropic.com/v1/"),
            api_key=api_key,
            sampling=sampling,
        )

    raise ValueError(f"unknown consumer: {kind!r} (expected openai_api | echo | claude_code)")
