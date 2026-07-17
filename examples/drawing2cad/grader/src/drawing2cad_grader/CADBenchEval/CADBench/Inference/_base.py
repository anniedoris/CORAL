from ..Processing import Processor
from typing import Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import multiprocessing as mp
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr


@contextmanager
def suppress_output_os():
    if sys.platform == "win32":
        yield
        return
    stdout_fd, stderr_fd = 1, 2
    saved_stdout_fd, saved_stderr_fd = os.dup(stdout_fd), os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

class InferenceEngine:
    def __init__(self,
                 processor: Processor,
                 num_workers: int = 16,
                 verbose: bool = True):
        self.processor = processor
        self.num_workers = num_workers
        self.verbose = verbose

    def _run_inference(self, input_data: Any):
        pass

    def _process_item(self, idx: int):
        try:
            with suppress_output_os(), suppress_output():
                input_data, file_id = self.processor[idx]
                output = self._run_inference(input_data)
            return {"file_id": file_id, **output}
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            detail = getattr(exc, "message", None) or str(exc)
            raise RuntimeError(
                f"Inference failed for idx={idx} (status={status_code}): {detail}"
            ) from None

    def _load_done(self, output_path: str):
        """Read an existing JSONL sink and index its rows by file_id."""
        done = {}
        if not (output_path and os.path.exists(output_path)):
            return done
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fid = row.get("file_id")
                if fid is not None:
                    done[str(fid)] = row
        return done

    def run(self, output_path: Optional[str] = None, resume: bool = True):
        # Map every item index to its file_id up front (cheap column read, no
        # image decoding) so we can skip already-completed rows on resume.
        try:
            all_ids = [str(x) for x in self.processor.dataset[self.processor.id_field]]
        except (AttributeError, KeyError, TypeError):
            all_ids = None

        done = self._load_done(output_path) if resume else {}
        if all_ids is not None:
            todo = [idx for idx in range(len(self.processor))
                    if all_ids[idx] not in done]
        else:
            todo = list(range(len(self.processor)))

        results = list(done.values())
        skipped = len(self.processor) - len(todo)
        if skipped and self.verbose:
            print(f"Resuming: {skipped} already done, {len(todo)} remaining.",
                  file=sys.stderr)

        # Append new rows as each future completes so progress survives a crash.
        # Single writer (this process), so no locking is needed.
        sink = None
        if output_path:
            sink = open(output_path, "a" if done else "w")
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp.get_context('fork')) as executor:
                futures = {executor.submit(self._process_item, idx): idx for idx in todo}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Running inference", disable=not self.verbose):
                    row = future.result()
                    results.append(row)
                    if sink is not None:
                        sink.write(json.dumps(row) + "\n")
                        sink.flush()
        finally:
            if sink is not None:
                sink.close()
        return results
