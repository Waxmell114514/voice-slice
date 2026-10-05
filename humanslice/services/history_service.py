from __future__ import annotations

from dataclasses import dataclass

from humanslice.models.project import ProjectData


@dataclass(slots=True)
class HistorySnapshot:
    """Serializable editor state used by undo / redo."""

    project: ProjectData
    current_segment_index: int | None
    selected_cutpoint_index: int | None


class HistoryManager:
    """Store undo / redo snapshots for project editing operations."""

    def __init__(self, max_depth: int = 100) -> None:
        self._max_depth = max(1, max_depth)
        self._undo_stack: list[HistorySnapshot] = []
        self._redo_stack: list[HistorySnapshot] = []

    def clear(self) -> None:
        """Remove all undo / redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def capture(
        self,
        project: ProjectData,
        current_segment_index: int | None,
        selected_cutpoint_index: int | None,
    ) -> HistorySnapshot:
        """Clone the current editor state into a snapshot."""
        return HistorySnapshot(
            project=ProjectData.from_dict(project.to_dict()),
            current_segment_index=current_segment_index,
            selected_cutpoint_index=selected_cutpoint_index,
        )

    def push_undo(self, snapshot: HistorySnapshot) -> None:
        """Push a pre-change snapshot and clear redo history."""
        self._undo_stack.append(self._clone_snapshot(snapshot))
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack = self._undo_stack[-self._max_depth :]
        self._redo_stack.clear()

    def undo(self, current_snapshot: HistorySnapshot) -> HistorySnapshot | None:
        """Return the previous snapshot and move current state to redo."""
        if not self._undo_stack:
            return None
        self._redo_stack.append(self._clone_snapshot(current_snapshot))
        return self._undo_stack.pop()

    def redo(self, current_snapshot: HistorySnapshot) -> HistorySnapshot | None:
        """Return the next snapshot and move current state back to undo."""
        if not self._redo_stack:
            return None
        self._undo_stack.append(self._clone_snapshot(current_snapshot))
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack = self._undo_stack[-self._max_depth :]
        return self._redo_stack.pop()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @staticmethod
    def states_differ(left: HistorySnapshot, right: HistorySnapshot) -> bool:
        """Compare two snapshots by project payload and selection state."""
        return (
            left.current_segment_index != right.current_segment_index
            or left.selected_cutpoint_index != right.selected_cutpoint_index
            or left.project.to_dict() != right.project.to_dict()
        )

    @staticmethod
    def _clone_snapshot(snapshot: HistorySnapshot) -> HistorySnapshot:
        return HistorySnapshot(
            project=ProjectData.from_dict(snapshot.project.to_dict()),
            current_segment_index=snapshot.current_segment_index,
            selected_cutpoint_index=snapshot.selected_cutpoint_index,
        )
