from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, QThread, Qt
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from humanslice.analysis.cutpoints import generate_candidate_cutpoints, sanitize_cut_points
from humanslice.app.f0_worker import F0PreviewWorker
from humanslice.audio.io import AudioLoadError, load_audio_file
from humanslice.audio.playback import PlaybackController
from humanslice.models.project import AudioTrack, ProjectData, Segment
from humanslice.services import segment_service
from humanslice.services.exporter import export_segments
from humanslice.services.history_service import HistoryManager, HistorySnapshot
from humanslice.services.project_service import load_project, save_project
from humanslice.services.settings_service import AppSettings
from humanslice.ui.progress import create_progress_dialog, translate_progress_message, update_progress
from humanslice.ui.segment_panel import SegmentDetailsPanel
from humanslice.ui.waveform_view import WaveformEditor

# (description, ((key sequence, handler), ...)) -- drives both QShortcut setup and the help panel.
ShortcutTable = tuple[tuple[str, tuple[tuple[str, Callable[[], object]], ...]], ...]


class MainWindow(QMainWindow):
    """HumanSlice desktop application main window."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.track: AudioTrack | None = None
        self.project = ProjectData(settings=settings.to_dict())
        self.current_project_path: Path | None = None
        self.current_segment_index: int | None = None
        self.selected_cutpoint_index: int | None = None
        self._segment_editor_sync = False
        self._shortcuts: list[QShortcut] = []
        self._f0_thread: QThread | None = None
        self._f0_job_id = 0
        self._f0_preview_times = None
        self._f0_preview_values = None
        self._cutpoint_insert_mode = False
        self.history = HistoryManager()

        self.playback = PlaybackController(volume=self.settings.playback.preview_volume, parent=self)

        self.setWindowTitle("HumanSlice")
        self.resize(1460, 860)
        self.setStatusBar(QStatusBar(self))

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._reset_ui_state()

    # ------------------------------------------------------------------ UI setup

    def _shortcut_table(self) -> ShortcutTable:
        return (
            ("导入音频", (("Ctrl+O", self.open_audio),)),
            ("打开工程", (("Ctrl+Shift+O", self.open_project),)),
            ("保存工程", (("Ctrl+S", self.save_current_project),)),
            ("撤销", (("Ctrl+Z", self.undo_last_change),)),
            ("重做", (("Ctrl+Shift+Z", self.redo_last_change), ("Ctrl+Y", self.redo_last_change))),
            ("自动切分", (("Ctrl+R", self.auto_segment),)),
            ("导出", (("Ctrl+E", self.export_current_segments),)),
            ("播放当前片段", (("Space", self._play_shortcut_handler),)),
            ("播放整段", (("Shift+Space", self._play_full_shortcut_handler),)),
            ("取消加点 / 停止播放", (("Escape", self._escape_shortcut_handler),)),
            ("删除选中切点", (("Delete", self._delete_shortcut_handler),)),
            (
                "切换片段",
                (
                    ("Alt+Left", lambda: self._select_relative_segment(-1)),
                    ("Alt+Right", lambda: self._select_relative_segment(1)),
                ),
            ),
            ("添加切点", (("Ctrl+D", self.toggle_cutpoint_insert_mode),)),
        )

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.setChildrenCollapsible(False)

        left_panel = QWidget(splitter)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_title = QLabel("文件 / 片段")
        self.file_label = QLabel("未加载音频")
        self.file_label.setWordWrap(True)
        self.audio_meta_label = QLabel("时长: -    采样率: -")
        self.segment_list = QListWidget()
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.audio_meta_label)
        left_layout.addWidget(self.segment_list, stretch=1)

        self.waveform_view = WaveformEditor(splitter)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.segment_panel = SegmentDetailsPanel()

        edit_group = QGroupBox("标签")
        edit_form = QFormLayout(edit_group)
        self.alias_input = QLineEdit()
        self.alias_input.setPlaceholderText("例如 a / shi / ka")
        self.notes_input = QPlainTextEdit()
        self.notes_input.setPlaceholderText("备注、发音提示、重采样说明等")
        self.notes_input.setTabChangesFocus(True)
        self.notes_input.setMinimumHeight(120)
        self.notes_input.installEventFilter(self)
        edit_form.addRow("Alias", self.alias_input)
        edit_form.addRow("备注", self.notes_input)

        shortcut_group = QGroupBox("快捷键")
        shortcut_layout = QVBoxLayout(shortcut_group)
        shortcut_layout.setContentsMargins(10, 10, 10, 10)
        shortcut_layout.setSpacing(4)
        for description, bindings in self._shortcut_table():
            keys = " / ".join(key for key, _ in bindings)
            label = QLabel(f"{keys} {description}")
            label.setWordWrap(True)
            shortcut_layout.addWidget(label)
        shortcut_layout.addStretch(1)

        right_layout.addWidget(self.segment_panel)
        right_layout.addWidget(edit_group)
        right_layout.addWidget(shortcut_group, stretch=1)

        splitter.addWidget(left_panel)
        splitter.addWidget(self.waveform_view)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)

        controls = QWidget(central)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        row1 = QHBoxLayout()
        self.open_audio_button = QPushButton("导入音频")
        self.open_project_button = QPushButton("打开工程")
        self.save_project_button = QPushButton("保存工程")
        self.undo_button = QPushButton("撤销")
        self.redo_button = QPushButton("重做")
        self.auto_segment_button = QPushButton("自动切分")
        self.export_button = QPushButton("导出")
        row1.addWidget(self.open_audio_button)
        row1.addWidget(self.open_project_button)
        row1.addWidget(self.save_project_button)
        row1.addWidget(self.undo_button)
        row1.addWidget(self.redo_button)
        row1.addWidget(self.auto_segment_button)
        row1.addStretch(1)
        row1.addWidget(self.export_button)

        row2 = QHBoxLayout()
        self.play_full_button = QPushButton("播放整段")
        self.play_segment_button = QPushButton("播放当前片段")
        self.play_boundary_button = QPushButton("播放边界窗口")
        self.play_transition_button = QPushButton("播放相邻拼接")
        self.insert_cutpoint_button = QPushButton("添加切点")
        self.insert_cutpoint_button.setCheckable(True)
        self.delete_cut_button = QPushButton("删除选中切点")
        self.stop_button = QPushButton("停止")
        row2.addWidget(self.play_full_button)
        row2.addWidget(self.play_segment_button)
        row2.addWidget(self.play_boundary_button)
        row2.addWidget(self.play_transition_button)
        row2.addWidget(self.insert_cutpoint_button)
        row2.addWidget(self.delete_cut_button)
        row2.addStretch(1)
        row2.addWidget(self.stop_button)

        controls_layout.addLayout(row1)
        controls_layout.addLayout(row2)

        root_layout.addWidget(splitter, stretch=1)
        root_layout.addWidget(controls)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.open_audio_button.clicked.connect(self.open_audio)
        self.open_project_button.clicked.connect(self.open_project)
        self.save_project_button.clicked.connect(self.save_current_project)
        self.undo_button.clicked.connect(self.undo_last_change)
        self.redo_button.clicked.connect(self.redo_last_change)
        self.auto_segment_button.clicked.connect(self.auto_segment)
        self.export_button.clicked.connect(self.export_current_segments)
        self.play_full_button.clicked.connect(self.play_full_track)
        self.play_segment_button.clicked.connect(self.play_current_segment)
        self.play_boundary_button.clicked.connect(self.play_boundary_window)
        self.play_transition_button.clicked.connect(self.play_transition_preview)
        self.insert_cutpoint_button.clicked.connect(self.toggle_cutpoint_insert_mode)
        self.delete_cut_button.clicked.connect(self.delete_selected_cutpoint)
        self.stop_button.clicked.connect(self.playback.stop)

        self.segment_list.currentRowChanged.connect(self._on_segment_chosen)
        self.waveform_view.segment_selected.connect(self._on_segment_chosen)
        self.waveform_view.add_cut_requested.connect(self.add_cutpoint_at_time)
        self.waveform_view.cutpoint_selected.connect(self._on_cutpoint_selected)
        self.waveform_view.cutpoint_moved.connect(self._on_cutpoint_moved)

        self.alias_input.editingFinished.connect(self._apply_alias_edit)

        self.playback.error_occurred.connect(self._show_error)
        self.playback.status_changed.connect(self.statusBar().showMessage)

    def _setup_shortcuts(self) -> None:
        for _, bindings in self._shortcut_table():
            for key_sequence, handler in bindings:
                shortcut = QShortcut(QKeySequence(key_sequence), self)
                shortcut.activated.connect(handler)
                self._shortcuts.append(shortcut)

    def _reset_ui_state(self) -> None:
        self._set_cutpoint_insert_mode(False)
        self.file_label.setText("未加载音频")
        self.audio_meta_label.setText("时长: -    采样率: -")
        self.segment_list.clear()
        self.waveform_view.clear()
        self._show_segment_details(None)
        self._update_history_actions()

    # ------------------------------------------------------------ file actions

    def open_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "Audio Files (*.wav *.mp3 *.flac)",
        )
        if path:
            self._load_audio_from_path(path)

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "Project Files (*.json)")
        if not path:
            return

        try:
            project = load_project(path)
        except Exception as exc:
            self._show_error(f"打开工程失败: {exc}")
            return

        if not project.audio_path:
            self._show_error("工程文件中没有音频路径。")
            return

        if self._load_audio_from_path(project.audio_path, project):
            self.current_project_path = Path(path)
            self.statusBar().showMessage(f"已打开工程: {self.current_project_path.name}", 4000)

    def save_current_project(self) -> None:
        self._commit_pending_text_edits()
        if not self._require_track():
            return

        target = self.current_project_path
        if target is None:
            path, _ = QFileDialog.getSaveFileName(self, "保存工程", "project.json", "Project Files (*.json)")
            if not path:
                return
            target = Path(path)

        self._stamp_project_metadata()
        try:
            saved_path = save_project(self.project, target)
        except Exception as exc:
            self._show_error(f"保存工程失败: {exc}")
            return

        self.current_project_path = saved_path
        self.statusBar().showMessage(f"工程已保存: {saved_path.name}", 4000)

    def auto_segment(self) -> None:
        self._commit_pending_text_edits()
        if not self._require_track():
            return

        before_snapshot = self._capture_history_snapshot()
        progress = create_progress_dialog(self, "自动切分", "正在准备切分...")
        try:
            cut_points = generate_candidate_cutpoints(
                self.track.samples,
                self.track.sample_rate,
                self.settings.cutpoint,
                progress_callback=lambda value, message: update_progress(
                    progress,
                    min(74, value),
                    translate_progress_message(message),
                ),
            )
            self.project.cut_points = cut_points
            self._rebuild_segments(
                preferred_index=0,
                analyze=True,
                progress_dialog=progress,
                progress_start=76,
                progress_end=100,
            )
            self._push_history_if_changed(before_snapshot)
            self.statusBar().showMessage(f"自动切分完成，共 {len(self.project.segments)} 个片段。", 5000)
        except Exception as exc:
            self._show_error(f"自动切分失败: {exc}")
        finally:
            progress.close()

    def export_current_segments(self) -> None:
        self._commit_pending_text_edits()
        if self.track is None or not self.project.segments:
            self._show_error("没有可导出的片段。")
            return

        pending = segment_service.segments_needing_analysis(self.project.segments)
        progress: QProgressDialog | None = None
        if pending:
            progress = create_progress_dialog(self, "导出前分析", "正在补全片段分析...")
            try:
                self._analyze_segments(pending, progress, 0, 100)
            except Exception as exc:
                progress.close()
                self._show_error(f"片段分析失败: {exc}")
                return

        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", "")
        if progress is not None:
            progress.close()
        if not output_dir:
            return

        try:
            result = export_segments(self.track, self.project.segments, output_dir)
        except Exception as exc:
            self._show_error(f"导出失败: {exc}")
            return

        self.statusBar().showMessage(
            f"导出完成: {result['segments_dir']} / {result['metadata_json'].name} / {result['oto_ini'].name}",
            6000,
        )

    def _load_audio_from_path(self, path: str, project_data: ProjectData | None = None) -> bool:
        self._set_cutpoint_insert_mode(False)
        try:
            track = load_audio_file(path, target_sr=self.settings.target_sample_rate)
        except AudioLoadError as exc:
            self._show_error(str(exc))
            return False

        self.track = track
        if project_data is None:
            self.current_project_path = None
        self.project = project_data or ProjectData()
        self._stamp_project_metadata()
        self._f0_preview_times = None
        self._f0_preview_values = None
        self.history.clear()
        if not self.project.cut_points:
            self.project.cut_points = [0.0, track.duration]

        self._rebuild_segments(preferred_index=0, analyze=False)
        self.waveform_view.set_audio(
            track.samples,
            track.sample_rate,
            self.project.cut_points,
            min_gap_sec=self._min_gap_sec(),
        )
        self._refresh_track_labels()
        self._start_f0_preview_job()
        self._update_history_actions()
        self.statusBar().showMessage(f"已加载音频: {track.display_name}", 4000)
        return True

    def _stamp_project_metadata(self) -> None:
        """Keep the project's audio / settings fields in sync with the loaded track."""
        if self.track is None:
            return
        self.project.audio_path = self.track.path
        self.project.sample_rate = self.track.sample_rate
        self.project.settings = self.settings.to_dict()

    # ------------------------------------------------------------ undo / redo

    def undo_last_change(self) -> None:
        self._commit_pending_text_edits()
        if self.track is None:
            return
        snapshot = self.history.undo(self._capture_history_snapshot())
        if snapshot is None:
            self.statusBar().showMessage("没有可撤销的操作。", 2500)
            self._update_history_actions()
            return
        self._restore_history_snapshot(snapshot)
        self.statusBar().showMessage("已撤销上一步操作。", 2500)

    def redo_last_change(self) -> None:
        self._commit_pending_text_edits()
        if self.track is None:
            return
        snapshot = self.history.redo(self._capture_history_snapshot())
        if snapshot is None:
            self.statusBar().showMessage("没有可重做的操作。", 2500)
            self._update_history_actions()
            return
        self._restore_history_snapshot(snapshot)
        self.statusBar().showMessage("已重做上一步操作。", 2500)

    def _capture_history_snapshot(self) -> HistorySnapshot:
        return self.history.capture(
            self.project,
            self.current_segment_index,
            self.selected_cutpoint_index,
        )

    def _push_history_if_changed(self, before_snapshot: HistorySnapshot) -> None:
        after_snapshot = self._capture_history_snapshot()
        if self.history.states_differ(before_snapshot, after_snapshot):
            self.history.push_undo(before_snapshot)
        self._update_history_actions()

    def _restore_history_snapshot(self, snapshot: HistorySnapshot) -> None:
        if self.track is None:
            return
        self.project = ProjectData.from_dict(snapshot.project.to_dict())
        self._stamp_project_metadata()
        self.current_segment_index = snapshot.current_segment_index
        self.selected_cutpoint_index = snapshot.selected_cutpoint_index
        self._sync_editor_from_project()
        self._update_history_actions()

    def _sync_editor_from_project(self) -> None:
        if self.track is None:
            return

        self.project.cut_points = sanitize_cut_points(
            self.project.cut_points or [0.0, self.track.duration],
            self.track.duration,
            self._min_gap_sec(),
        )
        self.waveform_view.set_audio(
            self.track.samples,
            self.track.sample_rate,
            self.project.cut_points,
            f0_times=self._f0_preview_times,
            f0_values=self._f0_preview_values,
            min_gap_sec=self._min_gap_sec(),
        )
        self._refresh_segment_list()

        if self.selected_cutpoint_index is not None and not self._is_inner_cutpoint(self.selected_cutpoint_index):
            self.selected_cutpoint_index = None
        self.waveform_view.select_cutpoint(self.selected_cutpoint_index)

        if self.project.segments:
            target_index = 0 if self.current_segment_index is None else self.current_segment_index
            self._select_segment(max(0, min(target_index, len(self.project.segments) - 1)))
        else:
            self._select_segment(-1)

    def _update_history_actions(self) -> None:
        has_track = self.track is not None
        self.undo_button.setEnabled(has_track and self.history.can_undo())
        self.redo_button.setEnabled(has_track and self.history.can_redo())

    # ------------------------------------------------------- segments / analysis

    def _rebuild_segments(
        self,
        preferred_index: int | None = None,
        analyze: bool = True,
        progress_dialog: QProgressDialog | None = None,
        progress_start: int = 0,
        progress_end: int = 100,
    ) -> None:
        if self.track is None:
            return

        self.project.cut_points = sanitize_cut_points(
            self.project.cut_points,
            self.track.duration,
            self._min_gap_sec(),
        )
        self.project.segments = segment_service.rebuild_segments(
            self.project.cut_points,
            self.track.duration,
            self.project.segments,
        )
        if analyze:
            pending = segment_service.segments_needing_analysis(self.project.segments)
            self._analyze_segments(pending, progress_dialog, progress_start, progress_end)

        self._refresh_segment_list()
        self.waveform_view.set_cutpoints(self.project.cut_points)
        target_index = preferred_index
        if target_index is None:
            target_index = self.current_segment_index or 0
        if self.project.segments:
            self._select_segment(max(0, min(target_index, len(self.project.segments) - 1)))
        else:
            self._select_segment(-1)

    def _analyze_segments(
        self,
        segments: list[Segment],
        progress_dialog: QProgressDialog | None = None,
        progress_start: int = 0,
        progress_end: int = 100,
    ) -> None:
        if self.track is None or not segments:
            return

        def report(done: int, total: int) -> None:
            if progress_dialog is None:
                return
            span = max(progress_end - progress_start, 1)
            update_progress(
                progress_dialog,
                progress_start + int(span * done / total),
                f"正在分析片段 {done}/{total}...",
            )

        segment_service.analyze_segments(
            self.track,
            segments,
            self.settings.regions,
            self.settings.scoring,
            progress_callback=report,
        )

    def _refresh_track_labels(self) -> None:
        if self.track is None:
            self.file_label.setText("未加载音频")
            self.audio_meta_label.setText("时长: -    采样率: -")
            return

        self.file_label.setText(self.track.display_name)
        self.audio_meta_label.setText(
            f"时长: {self.track.duration:.2f} s    采样率: {self.track.sample_rate} Hz    片段数: {len(self.project.segments)}"
        )

    def _refresh_segment_list(self) -> None:
        with QSignalBlocker(self.segment_list):
            self.segment_list.clear()
            for segment in self.project.segments:
                self.segment_list.addItem(QListWidgetItem(_segment_list_title(segment)))
        self._refresh_track_labels()

    def _select_segment(self, index: int) -> None:
        if index < 0 or index >= len(self.project.segments):
            self.current_segment_index = None
            self.waveform_view.set_current_segment(None)
            self._show_segment_details(None)
            return

        self.current_segment_index = index
        with QSignalBlocker(self.segment_list):
            self.segment_list.setCurrentRow(index)
        self.waveform_view.set_current_segment(self.project.segments[index])
        self._show_segment_details(self.project.segments[index])

    def _on_segment_chosen(self, index: int) -> None:
        self._clear_cutpoint_selection()
        self._select_segment(index)

    def _select_relative_segment(self, delta: int) -> None:
        if not self.project.segments:
            return
        base = self.current_segment_index or 0
        self._clear_cutpoint_selection()
        self._select_segment(max(0, min(base + delta, len(self.project.segments) - 1)))

    def _current_segment(self) -> Segment | None:
        if self.current_segment_index is None:
            return None
        if self.current_segment_index < 0 or self.current_segment_index >= len(self.project.segments):
            return None
        return self.project.segments[self.current_segment_index]

    # ------------------------------------------------------------ alias / notes

    def _show_segment_details(self, segment: Segment | None) -> None:
        self.segment_panel.show_segment(segment)
        self._segment_editor_sync = True
        try:
            with QSignalBlocker(self.alias_input), QSignalBlocker(self.notes_input):
                self.alias_input.setText("" if segment is None else segment.alias)
                self.notes_input.setPlainText("" if segment is None else segment.notes)
        finally:
            self._segment_editor_sync = False

    def _commit_pending_text_edits(self) -> None:
        self._apply_alias_edit()
        self._apply_notes_edit()

    def _apply_alias_edit(self) -> None:
        if self._segment_editor_sync or self.current_segment_index is None:
            return
        before_snapshot = self._capture_history_snapshot()
        segment = self.project.segments[self.current_segment_index]
        new_alias = self.alias_input.text().strip()
        if segment.alias == new_alias:
            return
        segment.alias = new_alias
        self._refresh_segment_list()
        self._select_segment(self.current_segment_index)
        self._push_history_if_changed(before_snapshot)

    def _apply_notes_edit(self) -> None:
        if self._segment_editor_sync or self.current_segment_index is None:
            return
        before_snapshot = self._capture_history_snapshot()
        new_notes = self.notes_input.toPlainText().strip()
        segment = self.project.segments[self.current_segment_index]
        if segment.notes == new_notes:
            return
        segment.notes = new_notes
        self._push_history_if_changed(before_snapshot)

    # --------------------------------------------------------------- cutpoints

    def add_cutpoint_at_time(self, time_value: float) -> None:
        if self.track is None:
            return
        before_snapshot = self._capture_history_snapshot()
        updated = sanitize_cut_points(
            [*self.project.cut_points, float(time_value)],
            self.track.duration,
            self._min_gap_sec(),
        )
        if updated == self.project.cut_points:
            self.statusBar().showMessage("切点太靠近已有边界，未添加。", 3000)
            return
        self.project.cut_points = updated
        preferred_index = segment_service.segment_index_for_time(self.project.segments, time_value)
        inserted_index = segment_service.nearest_cutpoint_index(self.project.cut_points, time_value)
        self.selected_cutpoint_index = inserted_index
        self._rebuild_segments(preferred_index=preferred_index, analyze=True)
        self.waveform_view.select_cutpoint(inserted_index)
        self._set_cutpoint_insert_mode(False)
        self._push_history_if_changed(before_snapshot)
        self.statusBar().showMessage(f"已添加切点: {self.project.cut_points[inserted_index]:.3f} s", 3000)

    def toggle_cutpoint_insert_mode(self) -> None:
        if not self._require_track():
            return
        self._set_cutpoint_insert_mode(not self._cutpoint_insert_mode)
        if self._cutpoint_insert_mode:
            self.statusBar().showMessage("添加切点模式：在波形或频谱上单击要添加的位置，Esc 取消。", 5000)
        else:
            self.statusBar().showMessage("已退出添加切点模式。", 2500)

    def _set_cutpoint_insert_mode(self, enabled: bool) -> None:
        active = bool(enabled and self.track is not None)
        self._cutpoint_insert_mode = active
        self.waveform_view.set_cutpoint_insert_mode(active)
        self.insert_cutpoint_button.setChecked(active)

    def delete_selected_cutpoint(self) -> None:
        if self.track is None:
            return
        if self.selected_cutpoint_index is None or self.selected_cutpoint_index <= 0:
            self.statusBar().showMessage("请先点击一个切点，再删除。", 3000)
            return
        if not self._is_inner_cutpoint(self.selected_cutpoint_index):
            return
        before_snapshot = self._capture_history_snapshot()
        del self.project.cut_points[self.selected_cutpoint_index]
        target_index = max(0, self.selected_cutpoint_index - 1)
        self._clear_cutpoint_selection()
        self._rebuild_segments(preferred_index=target_index, analyze=True)
        self._push_history_if_changed(before_snapshot)

    def _on_cutpoint_selected(self, cut_index: int) -> None:
        self.selected_cutpoint_index = cut_index
        self.waveform_view.select_cutpoint(cut_index)
        if self.project.segments:
            self._select_segment(max(0, min(cut_index - 1, len(self.project.segments) - 1)))

    def _on_cutpoint_moved(self, cut_index: int, time_value: float) -> None:
        if self.track is None or not self._is_inner_cutpoint(cut_index):
            return
        if abs(float(self.project.cut_points[cut_index]) - float(time_value)) <= 1e-6:
            return
        before_snapshot = self._capture_history_snapshot()
        self.project.cut_points[cut_index] = float(time_value)
        self.selected_cutpoint_index = cut_index
        self._rebuild_segments(preferred_index=max(0, cut_index - 1), analyze=True)
        self.waveform_view.select_cutpoint(cut_index)
        self._push_history_if_changed(before_snapshot)

    def _clear_cutpoint_selection(self) -> None:
        self.selected_cutpoint_index = None
        self.waveform_view.select_cutpoint(None)

    def _is_inner_cutpoint(self, cut_index: int) -> bool:
        """True for movable / deletable cutpoints, i.e. not the 0 or duration bounds."""
        return 0 < cut_index < len(self.project.cut_points) - 1

    def _min_gap_sec(self) -> float:
        return segment_service.min_gap_sec(self.settings.cutpoint)

    # ---------------------------------------------------------------- playback

    def play_full_track(self) -> None:
        if not self._require_track():
            return
        self.playback.play_full_track(self.track)

    def play_current_segment(self) -> None:
        if not self._require_track():
            return
        segment = self._current_segment()
        if segment is None:
            self._show_error("请先选择片段。")
            return
        self.playback.play_segment(self.track, segment)

    def play_boundary_window(self) -> None:
        if not self._require_track():
            return
        boundary_time: float | None = None
        if self.selected_cutpoint_index is not None and self._is_inner_cutpoint(self.selected_cutpoint_index):
            boundary_time = self.project.cut_points[self.selected_cutpoint_index]
        elif self.current_segment_index is not None and self.current_segment_index < len(self.project.cut_points) - 1:
            boundary_time = self.project.cut_points[self.current_segment_index + 1]
        if boundary_time is None:
            self._show_error("请先选择一个切点或片段。")
            return
        self.playback.play_boundary_window(
            self.track,
            boundary_time,
            window_ms=self.settings.playback.boundary_window_ms,
        )

    def play_transition_preview(self) -> None:
        if not self._require_track():
            return
        if self.current_segment_index is None:
            self._show_error("请先选择片段。")
            return
        if self.current_segment_index >= len(self.project.segments) - 1:
            self._show_error("当前片段后面没有相邻片段。")
            return
        self.playback.play_transition_preview(
            self.track,
            self.project.segments[self.current_segment_index],
            self.project.segments[self.current_segment_index + 1],
            crossfade_ms=self.settings.playback.preview_crossfade_ms,
        )

    # --------------------------------------------------------------- shortcuts

    def _play_shortcut_handler(self) -> None:
        if not self._editor_has_text_focus():
            self.play_current_segment()

    def _play_full_shortcut_handler(self) -> None:
        if not self._editor_has_text_focus():
            self.play_full_track()

    def _delete_shortcut_handler(self) -> None:
        if not self._editor_has_text_focus():
            self.delete_selected_cutpoint()

    def _escape_shortcut_handler(self) -> None:
        if self._cutpoint_insert_mode:
            self._set_cutpoint_insert_mode(False)
            self.statusBar().showMessage("已取消添加切点模式。", 2500)
            return
        self.playback.stop()

    def _editor_has_text_focus(self) -> bool:
        return isinstance(QApplication.focusWidget(), (QLineEdit, QPlainTextEdit))

    # -------------------------------------------------------------- F0 preview

    def _start_f0_preview_job(self) -> None:
        if self.track is None:
            return
        self._f0_job_id += 1
        thread = QThread(self)
        worker = F0PreviewWorker(self._f0_job_id, self.track.samples, self.track.sample_rate, self.settings.scoring)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_f0_preview_ready)
        worker.failed.connect(self._handle_f0_preview_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._f0_thread = thread
        thread.start()
        self.statusBar().showMessage("正在生成 F0 预览...", 2500)

    def _handle_f0_preview_ready(self, job_id: int, times: object, values: object) -> None:
        if job_id != self._f0_job_id:
            return
        self._f0_preview_times = times
        self._f0_preview_values = values
        self.waveform_view.set_f0_preview(times, values)
        self.statusBar().showMessage("F0 预览已更新。", 2500)

    def _handle_f0_preview_failed(self, job_id: int, message: str) -> None:
        if job_id != self._f0_job_id:
            return
        self.statusBar().showMessage(f"F0 预览生成失败: {message}", 4000)

    # ------------------------------------------------------------------- misc

    def _require_track(self) -> bool:
        if self.track is None:
            self._show_error("请先加载音频。")
            return False
        return True

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "HumanSlice", message)
        self.statusBar().showMessage(message, 5000)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.notes_input and event.type() == QEvent.Type.FocusOut:
            self._apply_notes_edit()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.playback.cleanup()
        if self._f0_thread is not None and self._f0_thread.isRunning():
            self._f0_thread.quit()
            self._f0_thread.wait(500)
        super().closeEvent(event)


def _segment_list_title(segment: Segment) -> str:
    title = segment.segment_id
    if segment.alias.strip():
        title = f"{title} [{segment.alias.strip()}]"
    meta = f"{segment.duration:.3f}s"
    if segment.score is not None:
        meta = (
            f"{meta} | C {segment.score.clarity_score:.0f}"
            f" / S {segment.score.stability_score:.0f}"
            f" | {segment.score.recommended_role}"
        )
    return f"{title}  {meta}"
