"""
Collect manuscript-ready material produced by each step.

Every step records (1) key numbers that will be cited in the manuscript or the response
to reviewers and (2) draft Methods/Results text with those numbers filled in.

    results/manuscript/key_numbers.csv          one row per number, all steps
    results/manuscript/notes/NN_step.md         draft text and numbers of one step

Re-running a step replaces its own rows/notes, so the files are always current.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import project_path


class ManuscriptNotes:
    def __init__(self, cfg: dict, step: str, title: str):
        self.step, self.title = step, title
        self.root = project_path(cfg, "results") / "manuscript"
        (self.root / "notes").mkdir(parents=True, exist_ok=True)
        self.numbers, self.sections = [], []

    def number(self, key: str, value, unit: str = "", description: str = "", reviewer: str = ""):
        """Record one number. key must be unique within the step, e.g. 'tc_flagged_cell_months'."""
        if isinstance(value, float):
            value = round(value, 4)
        self.numbers.append(dict(step=self.step, key=key, value=value, unit=unit,
                                 description=description, reviewer_comment=reviewer))
        return value

    def text(self, heading: str, body: str):
        """Draft manuscript text (English) for Methods, Results or Supplementary."""
        self.sections.append((heading, body.strip()))

    def save(self):
        kn = self.root / "key_numbers.csv"
        new = pd.DataFrame(self.numbers)
        if kn.exists():
            old = pd.read_csv(kn)
            old = old[old["step"] != self.step]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values(["step", "key"]).to_csv(kn, index=False)

        lines = [f"# {self.step} - {self.title}",
                 f"_Generated automatically on {datetime.now():%Y-%m-%d %H:%M}. Do not edit by hand; "
                 f"edit the script and re-run._", ""]
        for heading, body in self.sections:
            lines += [f"## {heading}", "", body, ""]
        lines += ["## Key numbers", "", "| key | value | unit | description | reviewer comment |",
                  "|---|---|---|---|---|"]
        for n in self.numbers:
            lines.append(f"| {n['key']} | {n['value']} | {n['unit']} | {n['description']} | "
                         f"{n['reviewer_comment']} |")
        (self.root / "notes" / f"{self.step}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  manuscript notes: results/manuscript/notes/{self.step}.md "
              f"({len(self.numbers)} key numbers -> results/manuscript/key_numbers.csv)")
