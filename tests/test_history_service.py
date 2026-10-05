from __future__ import annotations

from humanslice.models.project import ProjectData, Segment
from humanslice.services.history_service import HistoryManager


def test_history_manager_undo_redo_roundtrip() -> None:
    history = HistoryManager()
    project = ProjectData(cut_points=[0.0, 1.0], segments=[Segment("seg_001", 0.0, 1.0)])

    before = history.capture(project, current_segment_index=0, selected_cutpoint_index=None)
    project.cut_points = [0.0, 0.5, 1.0]
    project.segments = [
        Segment("seg_001", 0.0, 0.5),
        Segment("seg_002", 0.5, 1.0),
    ]
    history.push_undo(before)

    undone = history.undo(history.capture(project, current_segment_index=1, selected_cutpoint_index=1))
    assert undone is not None
    assert undone.project.cut_points == [0.0, 1.0]
    assert len(undone.project.segments) == 1

    redone = history.redo(undone)
    assert redone is not None
    assert redone.project.cut_points == [0.0, 0.5, 1.0]
    assert len(redone.project.segments) == 2


def test_history_manager_clears_redo_after_new_operation() -> None:
    history = HistoryManager()
    project = ProjectData(cut_points=[0.0, 1.0], segments=[Segment("seg_001", 0.0, 1.0)])

    original = history.capture(project, current_segment_index=0, selected_cutpoint_index=None)
    project.cut_points = [0.0, 0.4, 1.0]
    history.push_undo(original)

    current = history.capture(project, current_segment_index=0, selected_cutpoint_index=1)
    undone = history.undo(current)
    assert undone is not None
    assert history.can_redo()

    project.cut_points = [0.0, 0.25, 1.0]
    history.push_undo(undone)

    assert not history.can_redo()
