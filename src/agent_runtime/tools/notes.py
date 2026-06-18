# src/agent_runtime/tools/notes.py
"""Read-only file tools for the notes directory inside sample_data/.

Tools:
  - read_note(filename)  -> content
  - list_notes()         -> list of filenames

Both tools enforce a path boundary: all access is restricted to sample_data/notes/.
"""
from pathlib import Path
from typing import Type

from pydantic import BaseModel, Field, field_validator

from agent_runtime.tools.base import BaseTool

# ── boundary constant ──────────────────────────────────────────

_NOTES_DIR = (Path("sample_data") / "notes").resolve()


def _resolve_note_path(filename: str) -> Path:
    """Resolve and boundary-check a note filename.

    Raises ValueError if the resolved path escapes sample_data/notes/.
    """
    target = (_NOTES_DIR / filename).resolve()
    if not target.is_relative_to(_NOTES_DIR):
        raise ValueError(
            f"Access denied: '{filename}' resolves outside allowed notes directory."
        )
    return target


# ── read_note ──────────────────────────────────────────────────

class ReadNoteInput(BaseModel):
    filename: str = Field(..., description="Name of the note file (e.g. 'forecasting_notes.txt').")

    @field_validator("filename")
    @classmethod
    def reject_traversal(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Note filenames must be simple names, not paths.")
        return v


class ReadNoteOutput(BaseModel):
    filename: str
    content: str
    exists: bool


class ReadNoteTool(BaseTool[ReadNoteInput, ReadNoteOutput]):
    @property
    def name(self) -> str:
        return "read_note"

    @property
    def description(self) -> str:
        return "Read the full contents of a note file by name."

    @property
    def input_model(self) -> Type[ReadNoteInput]:
        return ReadNoteInput

    @property
    def output_model(self) -> Type[ReadNoteOutput]:
        return ReadNoteOutput

    def run(self, args: ReadNoteInput) -> ReadNoteOutput:
        file_path = _resolve_note_path(args.filename)
        if not file_path.exists():
            return ReadNoteOutput(
                filename=args.filename,
                content="",
                exists=False,
            )
        content = file_path.read_text(encoding="utf-8")
        return ReadNoteOutput(
            filename=args.filename,
            content=content,
            exists=True,
        )


# ── list_notes ─────────────────────────────────────────────────

class ListNotesInput(BaseModel):
    """No arguments needed — listing is always scoped to sample_data/notes/."""
    pass


class ListNotesOutput(BaseModel):
    filenames: list[str]


class ListNotesTool(BaseTool[ListNotesInput, ListNotesOutput]):
    @property
    def name(self) -> str:
        return "list_notes"

    @property
    def description(self) -> str:
        return "List all available note filenames in the notes directory."

    @property
    def input_model(self) -> Type[ListNotesInput]:
        return ListNotesInput

    @property
    def output_model(self) -> Type[ListNotesOutput]:
        return ListNotesOutput

    def run(self, args: ListNotesInput) -> ListNotesOutput:
        _NOTES_DIR.mkdir(parents=True, exist_ok=True)
        names = sorted(
            p.name for p in _NOTES_DIR.iterdir() if p.is_file()
        )
        return ListNotesOutput(filenames=names)