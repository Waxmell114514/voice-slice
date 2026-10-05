from __future__ import annotations

from pathlib import Path

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

from analysis.cutpoints import build_segments, generate_candidate_cutpoints, sanitize_cut_points
from analysis.regions import analyze_segment_regions
from analysis.scoring import score_segment
from app.f0_worker import F0PreviewWorker
from audio.io import AudioLoadError, load_audio_file
from audio.playback import PlaybackController
from models.project import ProjectData, Segment
from services.exporter import export_segments
from services.history_service import HistoryManager, HistorySnapshot
from services.project_service import load_project, save_project
from services.settings_service import AppSettings
from ui.waveform_view import WaveformEditor


class MainWindow(QMainWindow):
    """HumanSlice desktop application main window."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.track = None
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

        segment_group = QGroupBox("当前片段")
        segment_form = QFormLayout(segment_group)
        self.segment_id_label = QLabel("-")
        self.start_label = QLabel("-")
        self.end_label = QLabel("-")
        self.duration_label = QLabel("-")
        self.clarity_label = QLabel("-")
        self.stability_label = QLabel("-")
        self.role_label = QLabel("-")
        self.onset_label = QLabel("-")
        self.nucleus_label = QLabel("-")
        self.tail_label = QLabel("-")
        segment_form.addRow("ID", self.segment_id_label)
        segment_form.addRow("开始", self.start_label)
        segment_form.addRow("结束", self.end_label)
        segment_form.addRow("时长", self.duration_label)
        segment_form.addRow("Clarity", self.clarity_label)
        segment_form.addRow("Stability", self.stability_label)
        segment_form.addRow("推荐角色", self.role_label)
        segment_form.addRow("Onset", self.onset_label)
        segment_form.addRow("Nucleus", self.nucleus_label)
        segment_form.addRow("Tail", self.tail_label)

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
        for text in (
            "Ctrl+O 导入音频",
            "Ctrl+Shift+O 打开工程",
            "Ctrl+S 保存工程",
            "Ctrl+Z 撤销",
            "Ctrl+Shift+Z / Ctrl+Y 重做",
            "Ctrl+R 自动切分",
            "Ctrl+E 导出",
            "Space 播放当前片段",
            "Shift+Space 播放整段",
            "Escape 取消加点 / 停止播放",
            "Delete 删除选中切点",
            "Alt+Left / Alt+Right 切换片段",
            "Ctrl+D 添加切点",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            shortcut_layout.addWidget(label)
        shortcut_layout.addStretch(1)

        right_layout.addWidget(segment_group)
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
        self.split_center_button = QPushButton("添加切点")
        self.split_center_button.setCheckable(True)
        self.delete_cut_button = QPushButton("删除选中切点")
        self.stop_button = QPushButton("停止")
        row2.addWidget(self.play_full_button)
        row2.addWidget(self.play_segment_button)
        row2.addWidget(self.play_boundary_button)
        row2.addWidget(self.play_transition_button)
        row2.addWidget(self.split_center_button)
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
        self.split_center_button.clicked.connect(self.toggle_cutpoint_insert_mode)
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
        self._add_shortcut("Ctrl+O", self.open_audio)
        self._add_shortcut("Ctrl+Shift+O", self.open_project)
        self._add_shortcut("Ctrl+S", self.save_current_project)
        self._add_shortcut("Ctrl+Z", self.undo_last_change)
        self._add_shortcut("Ctrl+Shift+Z", self.redo_last_change)
        self._add_shortcut("Ctrl+Y", self.redo_last_change)
        self._add_shortcut("Ctrl+R", self.auto_segment)
        self._add_shortcut("Ctrl+E", self.export_current_segments)
        self._add_shortcut("Space", self._play_shortcut_handler)
        self._add_shortcut("Shift+Space", self._play_full_shortcut_handler)
        self._add_shortcut("Escape", self._escape_shortcut_handler)
        self._add_shortcut("Delete", self._delete_shortcut_handler)
        self._add_shortcut("Alt+Left", lambda: self._select_relative_segment(-1))
        self._add_shortcut("Alt+Right", lambda: self._select_relative_segment(1))
        self._add_shortcut("Ctrl+D", self.toggle_cutpoint_insert_mode)

    def _add_shortcut(self, key_sequence: str, handler: object) -> None:
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
        if self.track is None:
            self._show_error("请先加载音频。")
            return

        target = self.current_project_path
        if target is None:
            path, _ = QFileDialog.getSaveFileName(self, "保存工程", "project.json", "Project Files (*.json)")
            if not path:
                return
            target = Path(path)

        self.project.audio_path = self.track.path
        self.project.sample_rate = self.track.sample_rate
        self.project.settings = self.settings.to_dict()
        try:
            saved_path = save_project(self.project, target)
        except Exception as exc:
            self._show_error(f"保存工程失败: {exc}")
            return

        self.current_project_path = saved_path
        self.statusBar().showMessage(f"工程已保存: {saved_path.name}", 4000)

    def auto_segment(self) -> None:
        self._commit_pending_text_edits()
        if self.track is None:
            self._show_error("请先加载音频。")
            return

        before_snapshot = self._capture_history_snapshot()
        progress = self._create_progress_dialog("自动切分", "正在准备切分...")
        try:
            cut_points = generate_candidate_cutpoints(
                self.track.samples,
                self.track.sample_rate,
                self.settings.cutpoint,
                progress_callback=lambda value, message: self._update_progress(
                    progress,
                    min(74, value),
                    self._translate_progress_message(message),
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

        pending = [segment for segment in self.project.segments if segment.regions is None or segment.score is None]
        progress: QProgressDialog | None = None
        if pending:
            progress = self._create_progress_dialog("导出前分析", "正在补全片段分析...")
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
        self.project.audio_path = self.track.path
        self.project.sample_rate = self.track.sample_rate
        self.project.settings = self.settings.to_dict()
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

        if self.selected_cutpoint_index is not None:
            if 0 < self.selected_cutpoint_index < len(self.project.cut_points) - 1:
                self.waveform_view.select_cutpoint(self.selected_cutpoint_index)
            else:
                self.selected_cutpoint_index = None
                self.waveform_view.select_cutpoint(None)
        else:
            self.waveform_view.select_cutpoint(None)

        if self.project.segments:
            target_index = 0 if self.current_segment_index is None else self.current_segment_index
            self._select_segment(max(0, min(target_index, len(self.project.segments) - 1)))
        else:
            self._select_segment(-1)

    def _update_history_actions(self) -> None:
        can_undo = self.history.can_undo() and self.track is not None
        can_redo = self.history.can_redo() and self.track is not None
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)

    def _commit_pending_text_edits(self) -> None:
        self._apply_alias_edit()
        self._apply_notes_edit()

    def _load_audio_from_path(self, path: str, project_data: ProjectData | None = None) -> bool:
        self._set_cutpoint_insert_mode(False)
        try:
            track = load_audio_file(path, target_sr=self.settings.target_sample_rate)
        except AudioLoadError as exc:
            self._show_error(str(exc))
            return False

        self.track = track
        self.current_project_path = None if project_data is None else self.current_project_path
        self.project = project_data or ProjectData(settings=self.settings.to_dict())
        self.project.audio_path = track.path
        self.project.sample_rate = track.sample_rate
        self.project.settings = self.settings.to_dict()
        self._f0_preview_times = None
        self._f0_preview_values = None
        self.history.clear()
        if not self.project.cut_points:
            self.project.cut_points = [0.0, track.duration]

        self.project.cut_points = sanitize_cut_points(
            self.project.cut_points,
            track.duration,
            self._min_gap_sec(),
        )
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
        self.statusBar().showMessage(f"已加载音频: {Path(track.path).name}", 4000)
        return True

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
        previous_segments = list(self.project.segments)
        segments = build_segments(self.project.cut_points, self.track.duration)
        pending_analysis: list[Segment] = []

        for segment in segments:
            previous = self._match_previous_segment(segment, previous_segments)
            if previous is not None:
                segment.alias = previous.alias
                segment.notes = previous.notes
                if self._same_bounds(segment, previous):
                    segment.regions = previous.regions
                    segment.score = previous.score
            if analyze and (segment.regions is None or segment.score is None):
                pending_analysis.append(segment)

        self.project.segments = segments
        if analyze and pending_analysis:
            self._analyze_segments(pending_analysis, progress_dialog, progress_start, progress_end)

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

        total = len(segments)
        span = max(progress_end - progress_start, 1)
        for index, segment in enumerate(segments, start=1):
            segment.regions = analyze_segment_regions(
                self.track.samples,
                self.track.sample_rate,
                segment.start,
                segment.end,
                self.settings.regions,
            )
            segment.score = score_segment(
                self.track.samples,
                self.track.sample_rate,
                segment,
                segment.regions,
                self.settings.scoring,
            )
            if progress_dialog is not None:
                progress_value = progress_start + int(span * index / total)
                self._update_progress(progress_dialog, progress_value, f"正在分析片段 {index}/{total}...")

    def _refresh_track_labels(self) -> None:
        if self.track is None:
            self.file_label.setText("未加载音频")
            self.audio_meta_label.setText("时长: -    采样率: -")
            return

        self.file_label.setText(Path(self.track.path).name)
        self.audio_meta_label.setText(
            f"时长: {self.track.duration:.2f} s    采样率: {self.track.sample_rate} Hz    片段数: {len(self.project.segments)}"
        )

    def _refresh_segment_list(self) -> None:
        blocker = QSignalBlocker(self.segment_list)
        self.segment_list.clear()
        for segment in self.project.segments:
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
            item = QListWidgetItem(f"{title}  {meta}")
            self.segment_list.addItem(item)
        del blocker
        self._refresh_track_labels()

    def _select_segment(self, index: int) -> None:
        if index < 0 or index >= len(self.project.segments):
            self.current_segment_index = None
            self.waveform_view.set_current_segment(None)
            self._show_segment_details(None)
            return

        self.current_segment_index = index
        blocker = QSignalBlocker(self.segment_list)
        self.segment_list.setCurrentRow(index)
        del blocker
        self.waveform_view.set_current_segment(self.project.segments[index])
        self._show_segment_details(self.project.segments[index])

    def _on_segment_chosen(self, index: int) -> None:
        self.selected_cutpoint_index = None
        self.waveform_view.select_cutpoint(None)
        self._select_segment(index)

    def _show_segment_details(self, segment: Segment | None) -> None:
        self._segment_editor_sync = True
        try:
            alias_blocker = QSignalBlocker(self.alias_input)
            notes_blocker = QSignalBlocker(self.notes_input)
            if segment is None:
                self.segment_id_label.setText("-")
                self.start_label.setText("-")
                self.end_label.setText("-")
                self.duration_label.setText("-")
                self.clarity_label.setText("-")
                self.stability_label.setText("-")
                self.role_label.setText("-")
                self.onset_label.setText("-")
                self.nucleus_label.setText("-")
                self.tail_label.setText("-")
                self.alias_input.clear()
                self.notes_input.clear()
            else:
                self.segment_id_label.setText(segment.segment_id)
                self.start_label.setText(f"{segment.start:.4f} s")
                self.end_label.setText(f"{segment.end:.4f} s")
                self.duration_label.setText(f"{segment.duration:.4f} s")
                if segment.score is None:
                    self.clarity_label.setText("-")
                    self.stability_label.setText("-")
                    self.role_label.setText("-")
                else:
                    self.clarity_label.setText(f"{segment.score.clarity_score:.1f}")
                    self.stability_label.setText(f"{segment.score.stability_score:.1f}")
                    self.role_label.setText(segment.score.recommended_role)
                self.onset_label.setText(self._format_region(segment.regions.onset if segment.regions else None))
                self.nucleus_label.setText(self._format_region(segment.regions.nucleus if segment.regions else None))
                self.tail_label.setText(self._format_region(segment.regions.tail if segment.regions else None))
                self.alias_input.setText(segment.alias)
                self.notes_input.setPlainText(segment.notes)
            del alias_blocker
            del notes_blocker
        finally:
            self._segment_editor_sync = False

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

    def add_cutpoint_at_time(self, time_value: float) -> None:
        if self.track is None:
            return
        before_snapshot = self._capture_history_snapshot()
        cut_points = list(self.project.cut_points)
        cut_points.append(float(time_value))
        updated = sanitize_cut_points(cut_points, self.track.duration, self._min_gap_sec())
        if updated == self.project.cut_points:
            self.statusBar().showMessage("切点太靠近已有边界，未添加。", 3000)
            return
        self.project.cut_points = updated
        preferred_index = max(0, self._segment_index_for_time(time_value))
        inserted_index = self._nearest_cutpoint_index(time_value)
        self.selected_cutpoint_index = inserted_index
        self._rebuild_segments(preferred_index=preferred_index, analyze=True)
        self.waveform_view.select_cutpoint(inserted_index)
        self._set_cutpoint_insert_mode(False)
        self._push_history_if_changed(before_snapshot)
        self.statusBar().showMessage(f"已添加切点: {self.project.cut_points[inserted_index]:.3f} s", 3000)

    def toggle_cutpoint_insert_mode(self) -> None:
        if self.track is None:
            self._show_error("请先加载音频。")
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
        self.split_center_button.setChecked(active)

    def delete_selected_cutpoint(self) -> None:
        if self.track is None:
            return
        if self.selected_cutpoint_index is None or self.selected_cutpoint_index <= 0:
            self.statusBar().showMessage("请先点击一个切点，再删除。", 3000)
            return
        if self.selected_cutpoint_index >= len(self.project.cut_points) - 1:
            return
        before_snapshot = self._capture_history_snapshot()
        del self.project.cut_points[self.selected_cutpoint_index]
        target_index = max(0, self.selected_cutpoint_index - 1)
        self.selected_cutpoint_index = None
        self.waveform_view.select_cutpoint(None)
        self._rebuild_segments(preferred_index=target_index, analyze=True)
        self._push_history_if_changed(before_snapshot)

    def _on_cutpoint_selected(self, cut_index: int) -> None:
        self.selected_cutpoint_index = cut_index
        self.waveform_view.select_cutpoint(cut_index)
        target_index = max(0, min(cut_index - 1, len(self.project.segments) - 1))
        if self.project.segments:
            self._select_segment(target_index)

    def _on_cutpoint_moved(self, cut_index: int, time_value: float) -> None:
        if self.track is None or cut_index <= 0 or cut_index >= len(self.project.cut_points) - 1:
            return
        current_value = float(self.project.cut_points[cut_index])
        if abs(current_value - float(time_value)) <= 1e-6:
            return
        before_snapshot = self._capture_history_snapshot()
        self.project.cut_points[cut_index] = float(time_value)
        self.selected_cutpoint_index = cut_index
        self._rebuild_segments(preferred_index=max(0, cut_index - 1), analyze=True)
        self.waveform_view.select_cutpoint(cut_index)
        self._push_history_if_changed(before_snapshot)

    def play_full_track(self) -> None:
        if self.track is None:
            self._show_error("请先加载音频。")
            return
        self.playback.play_full_track(self.track)

    def play_current_segment(self) -> None:
        if self.track is None:
            self._show_error("请先加载音频。")
            return
        segment = self._current_segment()
        if segment is None:
            self._show_error("请先选择片段。")
            return
        self.playback.play_segment(self.track, segment)

    def play_boundary_window(self) -> None:
        if self.track is None:
            self._show_error("请先加载音频。")
            return
        boundary_time: float | None = None
        if self.selected_cutpoint_index is not None and 0 < self.selected_cutpoint_index < len(self.project.cut_points) - 1:
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
        if self.track is None:
            self._show_error("请先加载音频。")
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

    def _select_relative_segment(self, delta: int) -> None:
        if not self.project.segments:
            return
        base = self.current_segment_index or 0
        target = max(0, min(base + delta, len(self.project.segments) - 1))
        self.selected_cutpoint_index = None
        self.waveform_view.select_cutpoint(None)
        self._select_segment(target)

    def _editor_has_text_focus(self) -> bool:
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QPlainTextEdit))

    def _current_segment(self) -> Segment | None:
        if self.current_segment_index is None:
            return None
        if self.current_segment_index < 0 or self.current_segment_index >= len(self.project.segments):
            return None
        return self.project.segments[self.current_segment_index]

    def _segment_index_for_time(self, time_value: float) -> int:
        for index, segment in enumerate(self.project.segments):
            if segment.start <= time_value <= segment.end:
                return index
        return max(0, len(self.project.segments) - 1)

    def _nearest_cutpoint_index(self, time_value: float) -> int:
        if not self.project.cut_points:
            return 0
        return min(
            range(len(self.project.cut_points)),
            key=lambda index: abs(self.project.cut_points[index] - time_value),
        )

    def _match_previous_segment(self, current: Segment, previous_segments: list[Segment]) -> Segment | None:
        best_segment: Segment | None = None
        best_ratio = 0.0
        for previous in previous_segments:
            overlap = max(0.0, min(current.end, previous.end) - max(current.start, previous.start))
            if overlap <= 0.0:
                continue
            ratio = overlap / max(current.duration, previous.duration, 1e-6)
            if ratio > best_ratio:
                best_ratio = ratio
                best_segment = previous
        return best_segment if best_ratio >= 0.55 else None

    def _same_bounds(self, left: Segment, right: Segment) -> bool:
        return abs(left.start - right.start) <= 0.002 and abs(left.end - right.end) <= 0.002

    def _start_f0_preview_job(self) -> None:
        if self.track is None:
            return
        self._f0_job_id += 1
        job_id = self._f0_job_id
        thread = QThread(self)
        worker = F0PreviewWorker(job_id, self.track.samples, self.track.sample_rate, self.settings.scoring)
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

    def _create_progress_dialog(self, title: str, label: str) -> QProgressDialog:
        dialog = QProgressDialog(label, "", 0, 100, self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setCancelButton(None)
        dialog.setValue(0)
        dialog.show()
        QApplication.processEvents()
        return dialog

    def _update_progress(self, dialog: QProgressDialog, value: int, message: str | None = None) -> None:
        if message:
            dialog.setLabelText(message)
        dialog.setValue(max(0, min(100, value)))
        QApplication.processEvents()

    def _translate_progress_message(self, message: str) -> str:
        mapping = {
            "Preparing audio for segmentation...": "正在准备音频...",
            "Computing onset and energy features...": "正在计算起音、能量与频谱特征...",
            "Combining segmentation cues...": "正在融合切分线索...",
            "Detecting onset and silence boundaries...": "正在检测起音与无声边界...",
            "Refining weak boundaries and breath fragments...": "正在合并弱边界与呼吸音碎片...",
            "Candidate cutpoints ready.": "候选切点已生成，正在分析片段...",
        }
        return mapping.get(message, message)

    def _format_region(self, region: object) -> str:
        if region is None:
            return "-"
        return f"{region.start:.4f} - {region.end:.4f} s"

    def _min_gap_sec(self) -> float:
        return max(0.02, self.settings.cutpoint.min_gap_ms / 1000.0)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.notes_input and event.type() == QEvent.Type.FocusOut:
            self._apply_notes_edit()
        return super().eventFilter(watched, event)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "HumanSlice", message)
        self.statusBar().showMessage(message, 5000)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.playback.cleanup()
        if self._f0_thread is not None and self._f0_thread.isRunning():
            self._f0_thread.quit()
            self._f0_thread.wait(500)
        super().closeEvent(event)
