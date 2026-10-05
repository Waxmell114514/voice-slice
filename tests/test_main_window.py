from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication, QLabel

from humanslice.app.main_window import MainWindow
from humanslice.services.settings_service import AppSettings


def test_main_window_cutpoint_edit_and_undo_roundtrip(tmp_path: Path) -> None:
    existing_app = QApplication.instance()
    app = existing_app or QApplication([])

    sample_rate = 22050
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, 0.5 * np.sin(2 * np.pi * 220.0 * t), sample_rate)

    window = MainWindow(settings=AppSettings())
    try:
        assert window._load_audio_from_path(str(audio_path))
        assert window.project.cut_points == [0.0, 1.0]
        assert len(window.project.segments) == 1
        assert window.segment_panel.segment_id_label.text() == "seg_001"

        window.add_cutpoint_at_time(0.4)
        assert window.project.cut_points == [0.0, 0.4, 1.0]
        assert len(window.project.segments) == 2
        assert all(segment.score is not None for segment in window.project.segments)
        assert window.selected_cutpoint_index == 1
        assert window.undo_button.isEnabled()

        window.undo_last_change()
        assert window.project.cut_points == [0.0, 1.0]
        window.redo_last_change()
        assert window.project.cut_points == [0.0, 0.4, 1.0]

        window.delete_selected_cutpoint()
        assert window.project.cut_points == [0.0, 1.0]

        window.segment_panel.show_segment(None)
        row_titles = {label.text() for label in window.segment_panel.findChildren(QLabel)}
        assert {"ID", "开始", "Tail"} <= row_titles
    finally:
        window.close()
        if existing_app is None:
            app.quit()


def test_make_dialog_validates_inputs(monkeypatch) -> None:
    existing_app = QApplication.instance()
    app = existing_app or QApplication([])
    from PySide6.QtWidgets import QMessageBox

    from humanslice.ui.make_dialog import MakeDialog

    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    dialog = MakeDialog()
    dialog.run_button.click()
    assert warnings and "素材" in warnings[0]
    assert dialog.run_button.isEnabled()
    dialog.close()
    if existing_app is None:
        app.quit()
