"""The test question set ("golden dataset") - eval/dataset.jsonl, one question per line.

    {
      "id": "leave-annual",
      "question": "How many days of annual leave do full-time employees get per year?",
      "answerable": true,                       # false = the documents can't answer it
      "tags": ["docx", "table"],                # used for the per-type breakdown
      "expected_sources": [                     # which chunk(s) contain the answer
        {"file": "employee_handbook.docx", "contains": "Annual leave | 25"}
      ],
      "must_include": [["25"]],                 # key facts; inner list = accepted spellings
      "reference": "Full-time employees get 25 days of paid annual leave per year."
    }

A chunk counts as a correct source when it comes from `file` AND its text
contains `contains` (case-insensitive). That is robust to re-chunking: chunk
ids change, but the text that holds the answer doesn't.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    id: str
    question: str
    answerable: bool
    tags: list[str] = field(default_factory=list)
    expected_sources: list[dict] = field(default_factory=list)
    must_include: list[list[str]] = field(default_factory=list)
    reference: str = ""


def load_dataset(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    cases, problems, seen = [], [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            row = json.loads(line)
            case = EvalCase(**row)
        except (json.JSONDecodeError, TypeError) as e:
            problems.append(f"line {number}: {e}")
            continue
        if case.id in seen:
            problems.append(f"line {number}: duplicate id '{case.id}'")
        seen.add(case.id)
        if case.answerable and not case.expected_sources:
            problems.append(f"line {number}: answerable question needs expected_sources")
        if case.answerable and not case.must_include:
            problems.append(f"line {number}: answerable question needs must_include facts")
        cases.append(case)
    if problems:
        raise ValueError("Problems in the dataset:\n  " + "\n  ".join(problems))
    if not cases:
        raise ValueError(f"No questions in {path}")
    return cases
