"""Session-local dialogue over the read-only catalogue search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from uuid import UUID

from .contracts import FilterPatch, SearchFilters
from .llm import LocalLLM
from .search import SearchEngine
from .workflow import DialogueResponse as DialogueResponse
from .workflow import build_workflow, run_workflow


@dataclass
class SessionState:
    filters: SearchFilters = field(default_factory=SearchFilters)
    semantic_query: str = ""
    last_successful_ids: list[UUID] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)


class SearchService:
    def __init__(self, engine: SearchEngine | None = None, llm: LocalLLM | None = None):
        self.engine = engine or SearchEngine()
        self.llm = llm or LocalLLM()
        self._workflow = build_workflow(self.engine, self.llm)

    def handle(
        self, text: str, session: SessionState, ui_changes: FilterPatch | None = None
    ) -> DialogueResponse:
        """Return a fresh response and apply session changes after the graph completes."""
        with session._lock:
            response, request, selected_ids = run_workflow(
                self._workflow,
                text,
                ui_changes,
                session.filters,
                session.semantic_query,
                session.last_successful_ids,
            )
            if request is not None:
                session.filters, session.semantic_query = request.filters, request.query
            if selected_ids is not None:
                session.last_successful_ids = selected_ids.copy()
            return response
