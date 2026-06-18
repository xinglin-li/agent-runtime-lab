# src/agent_runtime/tools/series.py
"""Read-only file tool for CSV data inside sample_data/series/.

Tool:
  - describe_csv(filename)  -> headers + row count
"""
import csv
from pathlib import Path
from typing import Type

from pydantic import BaseModel, Field, field_validator

from agent_runtime.tools.base import BaseTool

# ── boundary constant ──────────────────────────────────────────

_SERIES_DIR = (Path("sample_data") / "series").resolve()


def _resolve_series_path(filename: str) -> Path:
    """Resolve and boundary-check a series filename."""
    target = (_SERIES_DIR / filename).resolve()
    if not target.is_relative_to(_SERIES_DIR):
        raise ValueError(
            f"Access denied: '{filename}' resolves outside allowed series directory."
        )
    return target


# ── describe_csv ───────────────────────────────────────────────

class DescribeCsvInput(BaseModel):
    filename: str = Field(..., description="Name of the CSV file (e.g. 'monthly_sales.csv').")

    @field_validator("filename")
    @classmethod
    def reject_traversal(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Series filenames must be simple names, not paths.")
        return v


class DescribeCsvOutput(BaseModel):
    filename: str
    headers: list[str]
    row_count: int
    exists: bool


class DescribeCsvTool(BaseTool[DescribeCsvInput, DescribeCsvOutput]):
    @property
    def name(self) -> str:
        return "describe_csv"

    @property
    def description(self) -> str:
        return "Read CSV headers and count rows from a data file in sample_data/series/."

    @property
    def input_model(self) -> Type[DescribeCsvInput]:
        return DescribeCsvInput

    @property
    def output_model(self) -> Type[DescribeCsvOutput]:
        return DescribeCsvOutput

    def run(self, args: DescribeCsvInput) -> DescribeCsvOutput:
        file_path = _resolve_series_path(args.filename)
        if not file_path.exists():
            return DescribeCsvOutput(
                filename=args.filename,
                headers=[],
                row_count=0,
                exists=False,
            )
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            row_count = sum(1 for _ in reader)
        return DescribeCsvOutput(
            filename=args.filename,
            headers=headers,
            row_count=row_count,
            exists=True,
        )