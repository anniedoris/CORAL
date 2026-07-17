"""drawing2cad grader — orchestrates hints -> CAD synthesis -> IoU scoring.

For each sample in a fixed subset of the private test set:
  1. run the agent's drawing_processor on the drawing (isolated subprocess),
  2. feed the drawing + hints to the fixed consumer to get CadQuery code,
  3. collect {file_id, generated} for scoring.
Then score all outputs against the ground-truth STLs and return one number.

Per-sample consumer I/O (prompt, raw output, extracted code) is logged to
generations.jsonl for inspection.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from coral.grader import TaskGrader

from . import dataset, runner, scoring
from .consumer import GenerationResult, make_consumer


class Grader(TaskGrader):
    """Evaluate drawing2cad submissions (agent evolves solution.drawing_processor)."""

    def evaluate(self):
        args = self.args
        program_file = args.get("program_file", "solution.py")

        # 1. Load the private test set and pick the fixed scoring subset.
        testset_dir = Path(self.private_dir) / "taskdata" / "testset"
        try:
            samples = dataset.load_manifest(testset_dir)
        except FileNotFoundError as e:
            return self.fail(str(e))
        subset = dataset.pick_fixed_subset(samples, args.get("eval_subset"))
        if not subset:
            return self.fail(f"Test set under {testset_dir} is empty")

        # 2. Build the consumer (fails fast on missing API key / bad config).
        try:
            consumer = make_consumer(args)
        except Exception as e:
            return self.fail(f"Consumer init failed: {e}")

        processor_timeout = int(args.get("processor_timeout", 60))

        # 3. Per sample: drawing_processor -> hints -> consumer -> CadQuery code.
        outputs: list[dict] = []
        generations: list[dict] = []  # full consumer I/O, dumped to generations.jsonl
        scratch = Path(tempfile.mkdtemp(prefix="drawing2cad_"))
        try:
            for sample in subset:
                workdir = scratch / sample.file_id
                workdir.mkdir(parents=True, exist_ok=True)

                # Copy the drawing into the temp workdir and hand agent code ONLY
                # this path — never the real .coral/private/... path. Otherwise
                # drawing_processor (run here with grader privileges) could derive
                # the sibling ground-truth STL and read the answer. workdir lives
                # under /tmp, outside the private tree, so there is no route in.
                local_drawing = workdir / f"drawing{sample.image_path.suffix}"
                shutil.copy(sample.image_path, local_drawing)

                try:
                    hints = runner.run_processor(
                        self.codebase_path,
                        str(local_drawing),
                        str(workdir),
                        processor_timeout,
                        program_file=program_file,
                    )
                except runner.RunnerError as e:
                    # A broken processor => no hints; the consumer still runs on
                    # the raw drawing (the baseline path), just without help.
                    hints = {"text": [f"(drawing_processor error: {e})"], "images": []}

                try:
                    result = consumer.generate(
                        str(local_drawing), hints["text"], hints["images"]
                    )
                except Exception as e:
                    # A failed generation scores as an invalid solid (status 0).
                    result = GenerationResult(
                        code=f"# consumer error: {e}\n",
                        raw_response=str(e),
                        prompt={"error": str(e)},
                    )

                outputs.append({"file_id": sample.file_id, "generated": result.code})
                generations.append(
                    {
                        "file_id": sample.file_id,
                        "prompt": result.prompt,
                        "raw_response": result.raw_response,
                        "code": result.code,
                    }
                )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        gen_path = self._write_generations(generations)
        print(f"Grader: {len(outputs)} samples processed; generations -> {gen_path}")

        # 4. Score all outputs against ground-truth STLs (placeholder for now).
        breakdown = scoring.score_outputs(
            subset,
            outputs,
            cadbench_path=args.get("cadbench_path"),
            variable_name=args.get("variable_name", "result"),
            num_workers=int(args.get("eval_num_workers", 8)),
            timeout=int(args.get("code_exec_timeout", 30)),
        )

        explanation = (
            f"Aligned IoU (adj median)={breakdown.score:.4f} | "
            f"VSR={breakdown.vsr:.0f}% | n={breakdown.n}"
        )
        if breakdown.note:
            explanation += f" | {breakdown.note}"
        return self.score(breakdown.score, explanation)

    def _write_generations(self, records: list[dict]) -> Path:
        """Dump per-sample consumer I/O (prompt / raw output / code) to JSONL.

        Defaults to eval_logs_dir, which persists per attempt and is symlinked
        into each agent worktree at `.claude/eval_logs/` — so agents can inspect
        what the consumer produced from their hints. Override the location with
        grader.args.generations_dir (e.g. when running the local test harness).
        """
        override = self.args.get("generations_dir")
        out_dir = Path(override) if override else self.eval_logs_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "generations.jsonl"
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return path
