from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from models.project import RegionRange, Segment, SegmentRegions
from ui.waveform_view import WaveformEditor


class _MouseClickEvent:
    def __init__(self, scene_position: QPointF, *, double: bool = False) -> None:
        self._scene_position = scene_position
        self._double = double

    def button(self) -> Qt.MouseButton:
        return Qt.MouseButton.LeftButton

    def scenePos(self) -> QPointF:
        return self._scene_position

    def double(self) -> bool:
        return self._double


def test_waveform_editor_accepts_segment_region_overlays() -> None:
    existing_app = QApplication.instance()
    app = existing_app or QApplication([])
    editor = WaveformEditor()
    segment = Segment(
        segment_id="seg_001",
        start=0.0,
        end=0.30,
        regions=SegmentRegions(
            onset=RegionRange(0.0, 0.05),
            nucleus=RegionRange(0.05, 0.22),
            tail=RegionRange(0.22, 0.30),
        ),
    )

    sample_rate = 22050
    t = np.arange(int(0.30 * sample_rate), dtype=np.float32) / sample_rate
    samples = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    editor.set_audio(samples, sample_rate, [0.0, 0.15, 0.30])
    editor.set_current_segment(segment)

    waveform_x_range, waveform_y_range = editor.waveform_plot.getPlotItem().viewRange()
    f0_mouse_enabled = editor.f0_plot.getPlotItem().vb.state["mouseEnabled"]
    waveform_mouse_enabled = editor.waveform_plot.getPlotItem().vb.state["mouseEnabled"]

    assert editor._segment_region is not None
    assert editor._spectrogram_segment_region is not None
    assert len(editor._overlay_regions) == 3
    assert len(editor._spectrogram_overlay_regions) == 3
    assert len(editor._cut_lines) == 1
    assert len(editor._spectrogram_cut_lines) == 1
    assert waveform_x_range[0] <= 0.0
    assert waveform_x_range[1] >= 0.30
    assert waveform_y_range[0] < 0.0 < waveform_y_range[1]
    assert tuple(bool(value) for value in waveform_mouse_enabled) == (True, False)
    assert tuple(bool(value) for value in f0_mouse_enabled) == (True, False)
    editor.close()
    if existing_app is None:
        app.quit()


def test_waveform_editor_insert_mode_adds_cutpoint_at_clicked_time() -> None:
    existing_app = QApplication.instance()
    app = existing_app or QApplication([])
    editor = WaveformEditor()
    editor.resize(800, 520)

    sample_rate = 22050
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    captured: list[float] = []
    editor.add_cut_requested.connect(captured.append)
    editor.show()
    editor.set_audio(samples, sample_rate, [0.0, 1.0])
    editor.set_cutpoint_insert_mode(True)
    app.processEvents()

    view_box = editor.waveform_plot.getPlotItem().vb
    scene_position = view_box.mapViewToScene(QPointF(0.37, 0.0))
    editor._handle_plot_click(editor.waveform_plot, _MouseClickEvent(scene_position))

    assert captured
    assert abs(captured[0] - 0.37) < 0.01
    editor.close()
    if existing_app is None:
        app.quit()
