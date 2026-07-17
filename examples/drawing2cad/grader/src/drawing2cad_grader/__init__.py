"""drawing2cad grader package.

Pipeline: agent's drawing_processor -> hints -> CAD-writing consumer -> CadQuery
code -> CADBench Evaluator -> IoU vs ground-truth STL.

Module map:
  grader.py    orchestration (Grader.evaluate)
  dataset.py   load manifest.jsonl + build the Evaluator's dataset
  runner.py    subprocess-isolated drawing_processor invocation
  consumer.py  modular CAD-writing consumer (OpenAI API now, Claude Code later)
  scoring.py   CADBench Evaluator wrapper (placeholder for now)
"""
