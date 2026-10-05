from __future__ import annotations

import math
from functools import partial

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from models.project import Segment


class WaveformEditor(QWidget):
    """Waveform + spectrogram + F0 editor with linked timeline editing."""

    segment_selected = Signal(int)
    add_cut_requested = Signal(float)
    cutpoint_selected = Signal(int)
    cutpoint_moved = Signal(int, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)

        self._duration = 0.0
        self._sample_rate = 0
        self._cut_points: list[float] = []
        self._min_gap_sec = 0.08
        self._selected_cut_index: int | None = None
        self._cutpoint_insert_mode = False

        self.waveform_plot = pg.PlotWidget()
        self.spectrogram_plot = pg.PlotWidget()
        self.f0_plot = pg.PlotWidget()

        self._configure_plots()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.waveform_plot, stretch=5)
        layout.addWidget(self.spectrogram_plot, stretch=3)
        layout.addWidget(self.f0_plot, stretch=2)

        self._waveform_curve = self.waveform_plot.plot(pen=pg.mkPen("#202020", width=1))
        self._spectrogram_image = pg.ImageItem(axisOrder="row-major")
        if hasattr(self._spectrogram_image, "setAutoDownsample"):
            self._spectrogram_image.setAutoDownsample(True)
        self.spectrogram_plot.addItem(self._spectrogram_image)
        self._apply_spectrogram_colormap()
        self._f0_curve = self.f0_plot.plot(
            pen=pg.mkPen("#005f73", width=2),
            connect="finite",
        )

        self._segment_region: pg.LinearRegionItem | None = None
        self._overlay_regions: list[pg.LinearRegionItem] = []
        self._spectrogram_segment_region: pg.LinearRegionItem | None = None
        self._spectrogram_overlay_regions: list[pg.LinearRegionItem] = []
        self._cut_lines: dict[int, pg.InfiniteLine] = {}
        self._spectrogram_cut_lines: dict[int, pg.InfiniteLine] = {}

        self.waveform_plot.scene().sigMouseClicked.connect(partial(self._handle_plot_click, self.waveform_plot))
        self.spectrogram_plot.scene().sigMouseClicked.connect(partial(self._handle_plot_click, self.spectrogram_plot))

    def clear(self) -> None:
        """Reset plots and overlays."""
        self._duration = 0.0
        self._sample_rate = 0
        self._cut_points = []
        self._selected_cut_index = None
        self.set_cutpoint_insert_mode(False)
        self._waveform_curve.setData([], [])
        self._f0_curve.setData([], [])
        self._set_blank_spectrogram()
        self._reset_view_ranges()
        self._clear_cut_lines()
        self._clear_regions()
        self.waveform_plot.setTitle("Waveform")
        self.spectrogram_plot.setTitle("Spectrogram")
        self.f0_plot.setTitle("F0 Preview")

    def set_audio(
        self,
        samples: np.ndarray,
        sample_rate: int,
        cut_points: list[float],
        f0_times: np.ndarray | None = None,
        f0_values: np.ndarray | None = None,
        min_gap_sec: float = 0.08,
    ) -> None:
        """Load a new audio buffer into the editor."""
        self._sample_rate = sample_rate
        self._duration = float(samples.shape[0] / sample_rate) if sample_rate > 0 else 0.0
        self._min_gap_sec = min_gap_sec

        x_values, y_values = _downsample_waveform(samples, sample_rate)
        self._waveform_curve.setData(x_values, y_values)
        self._set_spectrogram_preview(samples, sample_rate)

        if f0_times is not None and f0_values is not None and f0_times.size > 0:
            self._f0_curve.setData(f0_times, f0_values)
        else:
            self._f0_curve.setData([], [])

        max_duration = max(self._duration, 0.1)
        self._set_waveform_vertical_range(y_values)
        self._set_f0_vertical_range(f0_values)
        self._set_full_time_range(max_duration)
        self.waveform_plot.setTitle("Waveform")
        self.spectrogram_plot.setTitle("Spectrogram")
        self.f0_plot.setTitle("F0 Preview")
        self.set_cutpoints(cut_points)

    def set_cutpoints(self, cut_points: list[float]) -> None:
        """Redraw vertical cut lines on both waveform and spectrogram views."""
        self._cut_points = cut_points[:]
        self._clear_cut_lines()
        if len(cut_points) <= 2:
            return

        for cut_index in range(1, len(cut_points) - 1):
            left_bound = cut_points[cut_index - 1] + self._min_gap_sec
            right_bound = cut_points[cut_index + 1] - self._min_gap_sec
            self._cut_lines[cut_index] = self._create_cut_line(
                self.waveform_plot,
                cut_index,
                cut_points[cut_index],
                left_bound,
                right_bound,
            )
            self._spectrogram_cut_lines[cut_index] = self._create_cut_line(
                self.spectrogram_plot,
                cut_index,
                cut_points[cut_index],
                left_bound,
                right_bound,
            )

        self.select_cutpoint(self._selected_cut_index)

    def set_cutpoint_insert_mode(self, enabled: bool) -> None:
        """Use the next timeline click as a cutpoint insertion request."""
        self._cutpoint_insert_mode = bool(enabled)
        widgets = (self, self.waveform_plot, self.spectrogram_plot)
        if self._cutpoint_insert_mode:
            for widget in widgets:
                widget.setCursor(Qt.CursorShape.CrossCursor)
        else:
            for widget in widgets:
                widget.unsetCursor()

    def set_f0_preview(self, f0_times: np.ndarray, f0_values: np.ndarray) -> None:
        """Update only the F0 preview without redrawing waveform or cut lines."""
        if f0_times.size > 0 and f0_values.size > 0:
            self._f0_curve.setData(f0_times, f0_values)
        else:
            self._f0_curve.setData([], [])
        self._set_f0_vertical_range(f0_values)

    def select_cutpoint(self, cut_index: int | None) -> None:
        """Highlight the currently selected cut line across linked views."""
        self._selected_cut_index = cut_index
        active_pen = pg.mkPen("#bb3e03", width=3)
        idle_pen = pg.mkPen("#ee9b00", width=2)
        for index, line in self._cut_lines.items():
            line.setPen(active_pen if index == cut_index else idle_pen)
        for index, line in self._spectrogram_cut_lines.items():
            line.setPen(active_pen if index == cut_index else idle_pen)

    def set_current_segment(self, segment: Segment | None) -> None:
        """Highlight the selected segment and its onset / nucleus / tail regions."""
        self._clear_regions()
        if segment is None:
            return

        self._segment_region = self._add_region(
            self.waveform_plot,
            segment.start,
            segment.end,
            color=(38, 70, 83, 45),
            z_value=-20,
        )
        self._spectrogram_segment_region = self._add_region(
            self.spectrogram_plot,
            segment.start,
            segment.end,
            color=(38, 70, 83, 35),
            z_value=20,
        )

        if segment.regions is None:
            return

        self._overlay_regions.append(
            self._add_region(
                self.waveform_plot,
                segment.regions.onset.start,
                segment.regions.onset.end,
                (238, 155, 0, 60),
                -15,
            )
        )
        self._overlay_regions.append(
            self._add_region(
                self.waveform_plot,
                segment.regions.nucleus.start,
                segment.regions.nucleus.end,
                (10, 147, 150, 45),
                -14,
            )
        )
        self._overlay_regions.append(
            self._add_region(
                self.waveform_plot,
                segment.regions.tail.start,
                segment.regions.tail.end,
                (174, 32, 18, 45),
                -13,
            )
        )

        self._spectrogram_overlay_regions.append(
            self._add_region(
                self.spectrogram_plot,
                segment.regions.onset.start,
                segment.regions.onset.end,
                (238, 155, 0, 45),
                25,
            )
        )
        self._spectrogram_overlay_regions.append(
            self._add_region(
                self.spectrogram_plot,
                segment.regions.nucleus.start,
                segment.regions.nucleus.end,
                (10, 147, 150, 35),
                26,
            )
        )
        self._spectrogram_overlay_regions.append(
            self._add_region(
                self.spectrogram_plot,
                segment.regions.tail.start,
                segment.regions.tail.end,
                (174, 32, 18, 40),
                27,
            )
        )

    def _configure_plots(self) -> None:
        for plot in (self.waveform_plot, self.spectrogram_plot, self.f0_plot):
            plot.setBackground("#f7f7f2")
            plot.showGrid(x=True, y=True, alpha=0.2)

        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setLabel("bottom", "Time", units="s")
        self.waveform_plot.setMouseEnabled(x=True, y=False)

        self.spectrogram_plot.setLabel("left", "Freq", units="Hz")
        self.spectrogram_plot.setLabel("bottom", "Time", units="s")
        self.spectrogram_plot.setMouseEnabled(x=True, y=False)
        self.spectrogram_plot.setMaximumHeight(240)
        self.spectrogram_plot.setXLink(self.waveform_plot)

        self.f0_plot.setLabel("left", "F0", units="Hz")
        self.f0_plot.setLabel("bottom", "Time", units="s")
        self.f0_plot.setMouseEnabled(x=True, y=False)
        self.f0_plot.setMaximumHeight(170)
        self.f0_plot.setXLink(self.waveform_plot)

    def _apply_spectrogram_colormap(self) -> None:
        try:
            color_map = pg.colormap.get("magma")
        except Exception:
            return
        self._spectrogram_image.setLookupTable(color_map.getLookupTable(0.0, 1.0, 256))

    def _set_spectrogram_preview(self, samples: np.ndarray, sample_rate: int) -> None:
        spectrum, max_freq = _compute_spectrogram_preview(samples, sample_rate)
        if spectrum.size == 0 or max_freq <= 0.0:
            self._set_blank_spectrogram()
            return

        level_min = float(np.percentile(spectrum, 18))
        level_max = float(np.percentile(spectrum, 99.4))
        if level_max <= level_min:
            level_max = level_min + 1.0

        self._spectrogram_image.setImage(spectrum, autoLevels=False, levels=(level_min, level_max))
        self._spectrogram_image.setRect(QRectF(0.0, 0.0, max(self._duration, 0.1), max_freq))
        self.spectrogram_plot.setYRange(0.0, max(max_freq, 400.0), padding=0.02)
        self.spectrogram_plot.setLimits(yMin=0.0, yMax=max(max_freq, 400.0))

    def _set_blank_spectrogram(self) -> None:
        blank = np.zeros((2, 2), dtype=np.float32)
        self._spectrogram_image.setImage(blank, autoLevels=False, levels=(0.0, 1.0))
        self._spectrogram_image.setRect(QRectF(0.0, 0.0, 0.1, 1.0))
        self.spectrogram_plot.setYRange(0.0, 1.0, padding=0.0)
        self.spectrogram_plot.setLimits(yMin=0.0, yMax=1.0)

    def _set_waveform_vertical_range(self, waveform_values: np.ndarray) -> None:
        amplitude = 1.0
        if waveform_values.size > 0:
            amplitude = max(0.25, float(np.max(np.abs(waveform_values))) * 1.08)
        self.waveform_plot.setYRange(-amplitude, amplitude, padding=0.0)
        self.waveform_plot.setLimits(yMin=-amplitude, yMax=amplitude)

    def _set_f0_vertical_range(self, f0_values: np.ndarray | None) -> None:
        valid = np.zeros(0, dtype=np.float32) if f0_values is None else np.asarray(f0_values, dtype=np.float32)
        valid = valid[np.isfinite(valid)]
        upper = 800.0
        if valid.size > 0:
            upper = max(300.0, float(np.max(valid)) * 1.15)
        self.f0_plot.setYRange(0.0, upper, padding=0.0)
        self.f0_plot.setLimits(yMin=0.0, yMax=upper)

    def _set_full_time_range(self, duration: float) -> None:
        for plot in (self.waveform_plot, self.spectrogram_plot, self.f0_plot):
            plot.setXRange(0.0, duration, padding=0.0)
            plot.setLimits(xMin=0.0, xMax=duration)

    def _reset_view_ranges(self) -> None:
        self.waveform_plot.setXRange(0.0, 0.1, padding=0.0)
        self.spectrogram_plot.setXRange(0.0, 0.1, padding=0.0)
        self.f0_plot.setXRange(0.0, 0.1, padding=0.0)
        self.waveform_plot.setYRange(-1.0, 1.0, padding=0.0)
        self.f0_plot.setYRange(0.0, 800.0, padding=0.0)
        self.waveform_plot.setLimits(xMin=0.0, xMax=0.1, yMin=-1.0, yMax=1.0)
        self.spectrogram_plot.setLimits(xMin=0.0, xMax=0.1, yMin=0.0, yMax=1.0)
        self.f0_plot.setLimits(xMin=0.0, xMax=0.1, yMin=0.0, yMax=800.0)

    def _create_cut_line(
        self,
        plot: pg.PlotWidget,
        cut_index: int,
        position: float,
        left_bound: float,
        right_bound: float,
    ) -> pg.InfiniteLine:
        line = pg.InfiniteLine(
            pos=position,
            angle=90,
            movable=True,
            pen=pg.mkPen("#ee9b00", width=2),
            hoverPen=pg.mkPen("#ca6702", width=3),
        )
        line.setBounds((left_bound, right_bound))
        line.sigClicked.connect(partial(self._handle_cut_line_clicked, cut_index))
        line.sigPositionChangeFinished.connect(partial(self._handle_cut_line_moved, cut_index))
        plot.addItem(line)
        return line

    def _add_region(
        self,
        plot: pg.PlotWidget,
        start: float,
        end: float,
        color: tuple[int, int, int, int],
        z_value: int,
    ) -> pg.LinearRegionItem:
        region = pg.LinearRegionItem(values=(start, end), movable=False, brush=pg.mkBrush(color))
        transparent_pen = pg.mkPen((0, 0, 0, 0))
        for line in region.lines:
            line.setPen(transparent_pen)
            line.setHoverPen(transparent_pen)
        region.setZValue(z_value)
        plot.addItem(region)
        return region

    def _clear_cut_lines(self) -> None:
        for line in self._cut_lines.values():
            self.waveform_plot.removeItem(line)
        for line in self._spectrogram_cut_lines.values():
            self.spectrogram_plot.removeItem(line)
        self._cut_lines.clear()
        self._spectrogram_cut_lines.clear()

    def _clear_regions(self) -> None:
        if self._segment_region is not None:
            self.waveform_plot.removeItem(self._segment_region)
            self._segment_region = None
        if self._spectrogram_segment_region is not None:
            self.spectrogram_plot.removeItem(self._spectrogram_segment_region)
            self._spectrogram_segment_region = None
        for region in self._overlay_regions:
            self.waveform_plot.removeItem(region)
        for region in self._spectrogram_overlay_regions:
            self.spectrogram_plot.removeItem(region)
        self._overlay_regions.clear()
        self._spectrogram_overlay_regions.clear()

    def _handle_plot_click(self, plot: pg.PlotWidget, event: object) -> None:
        if self._duration <= 0:
            return
        button = getattr(event, "button", lambda: None)()
        if button != Qt.MouseButton.LeftButton:
            return

        scene_position = getattr(event, "scenePos", lambda: None)()
        if scene_position is None:
            return

        view_box = plot.getPlotItem().vb
        if not view_box.sceneBoundingRect().contains(scene_position):
            return

        mouse_point = view_box.mapSceneToView(scene_position)
        time_value = float(np.clip(mouse_point.x(), 0.0, self._duration))
        if self._cutpoint_insert_mode or getattr(event, "double", lambda: False)():
            self.add_cut_requested.emit(time_value)
            return

        segment_index = self._segment_index_for_time(time_value)
        if segment_index is not None:
            self.segment_selected.emit(segment_index)

    def _handle_cut_line_clicked(self, cut_index: int, *_: object) -> None:
        self.select_cutpoint(cut_index)
        self.cutpoint_selected.emit(cut_index)

    def _handle_cut_line_moved(self, cut_index: int, line: pg.InfiniteLine) -> None:
        self.cutpoint_moved.emit(cut_index, float(line.value()))

    def _segment_index_for_time(self, time_value: float) -> int | None:
        for index, (start, end) in enumerate(zip(self._cut_points[:-1], self._cut_points[1:])):
            if start <= time_value <= end:
                return index
        return None


def _downsample_waveform(
    samples: np.ndarray,
    sample_rate: int,
    max_points: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_rate <= 0 or samples.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    if samples.size <= max_points:
        x_values = np.arange(samples.size, dtype=np.float32) / float(sample_rate)
        return x_values, samples.astype(np.float32, copy=False)

    step = int(np.ceil(samples.size / max_points))
    trimmed_size = (samples.size // step) * step
    trimmed = samples[:trimmed_size]
    if trimmed.size == 0:
        trimmed = samples
        step = max(1, samples.size)

    reshaped = trimmed.reshape(-1, step)
    x_values = (np.arange(reshaped.shape[0], dtype=np.float32) * step) / float(sample_rate)
    y_min = reshaped.min(axis=1)
    y_max = reshaped.max(axis=1)

    x_plot = np.repeat(x_values, 2)
    y_plot = np.empty(y_min.size * 2, dtype=np.float32)
    y_plot[0::2] = y_min
    y_plot[1::2] = y_max
    return x_plot, y_plot


def _compute_spectrogram_preview(
    samples: np.ndarray,
    sample_rate: int,
    preview_max_sample_rate: int = 12000,
    preview_max_frames: int = 1800,
    max_display_hz: float = 6000.0,
) -> tuple[np.ndarray, float]:
    """Build a lightweight spectrogram image for interactive preview."""
    if sample_rate <= 0 or samples.size < 128:
        return np.zeros((0, 0), dtype=np.float32), 0.0

    preview_samples = samples.astype(np.float32, copy=False)
    preview_rate = int(sample_rate)

    if preview_max_sample_rate > 0 and preview_rate > preview_max_sample_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(preview_rate, preview_max_sample_rate)
        up = preview_max_sample_rate // divisor
        down = preview_rate // divisor
        preview_samples = resample_poly(preview_samples, up, down).astype(np.float32, copy=False)
        preview_rate = int(preview_max_sample_rate)

    if preview_samples.size < 128:
        return np.zeros((0, 0), dtype=np.float32), 0.0

    from scipy import signal

    hop_length = max(64, int(np.ceil(preview_samples.size / max(preview_max_frames, 1))))
    nperseg = max(256, _next_power_of_two(hop_length * 4))
    nperseg = min(nperseg, 2048, preview_samples.size)
    if nperseg < 128:
        return np.zeros((0, 0), dtype=np.float32), 0.0

    hop_length = min(hop_length, max(32, nperseg - 32))
    noverlap = max(0, nperseg - hop_length)

    frequencies, _, spectrum = signal.spectrogram(
        preview_samples,
        fs=preview_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )

    if spectrum.size == 0 or frequencies.size == 0:
        return np.zeros((0, 0), dtype=np.float32), 0.0

    display_ceiling = min(float(preview_rate) / 2.0, max_display_hz)
    keep = frequencies <= display_ceiling
    if not np.any(keep):
        return np.zeros((0, 0), dtype=np.float32), 0.0

    clipped = spectrum[keep]
    spectrum_db = (20.0 * np.log10(np.maximum(clipped, 1e-6))).astype(np.float32, copy=False)
    return spectrum_db, float(frequencies[keep][-1])


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << int(np.ceil(np.log2(value)))
