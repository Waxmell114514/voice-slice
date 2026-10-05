"""“一键人力” dialog: material folder + song -> jinriki vocal, voicebank and USTX."""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from humanslice.pipeline import MakeOptions, run_make
from humanslice.render.engine import RenderOptions
from humanslice.render.select import SelectOptions


class MakeWorker(QObject):
    progress = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, options: MakeOptions) -> None:
        super().__init__()
        self._options = options

    @Slot()
    def run(self) -> None:
        try:
            results = run_make(self._options, progress=self.progress.emit)
        except Exception as exc:  # surfaced to the user in the dialog
            self.progress.emit(traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit({key: str(path) for key, path in results.items()})


class MakeDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("一键人力")
        self.resize(760, 640)
        self._thread: QThread | None = None
        self._results: dict[str, str] = {}

        self.material_input = QLineEdit()
        self.vocal_input = QLineEdit()
        self.lyrics_input = QLineEdit()
        self.lyrics_input.setPlaceholderText("歌词 .txt / .lrc，或直接粘贴歌词；留空则自动识别")
        self.output_input = QLineEdit(str(Path.cwd() / "out" / "jinriki"))
        self.name_input = QLineEdit("HumanSlice")
        self.separate_check = QCheckBox("歌曲是完整混音（自动分离人声与伴奏，并输出混音成品）")
        self.separate_material_check = QCheckBox("素材含背景音乐（先做人声分离，较慢）")
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("PC-NSF-HiFiGAN（音质最好，权重仅限非商用）", "nsf")
        self.backend_combo.addItem("WORLD（无许可限制）", "world")
        self.join_spin = QDoubleSpinBox()
        self.join_spin.setRange(0.0, 5.0)
        self.join_spin.setSingleStep(0.5)
        self.join_spin.setValue(1.0)
        self.join_spin.setToolTip("越大越偏好素材中原本连续的整词（人力感更强）")

        form = QFormLayout()
        form.addRow("素材文件夹", self._with_browse(self.material_input, self._browse_material))
        form.addRow("歌曲（清唱或完整歌曲）", self._with_browse(self.vocal_input, self._browse_vocal))
        form.addRow("歌词", self._with_browse(self.lyrics_input, self._browse_lyrics))
        form.addRow("输出文件夹", self._with_browse(self.output_input, self._browse_output))
        form.addRow("音源名称", self.name_input)
        form.addRow("", self.separate_check)
        form.addRow("", self.separate_material_check)
        form.addRow("声码器", self.backend_combo)
        form.addRow("原句连接偏好", self.join_spin)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.run_button = QPushButton("开始制作")
        self.open_mix_button = QPushButton("播放成品")
        self.open_folder_button = QPushButton("打开输出文件夹")
        for button in (self.open_mix_button, self.open_folder_button):
            button.setEnabled(False)
        self.run_button.clicked.connect(self._start)
        self.open_mix_button.clicked.connect(self._open_result)
        self.open_folder_button.clicked.connect(lambda: self._open_path(self.output_input.text()))

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addStretch(1)
        buttons.addWidget(self.open_mix_button)
        buttons.addWidget(self.open_folder_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.log, stretch=1)
        layout.addLayout(buttons)

    # ------------------------------------------------------------- widgets

    def _with_browse(self, line_edit: QLineEdit, handler) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("浏览…")
        button.clicked.connect(handler)
        layout.addWidget(line_edit, stretch=1)
        layout.addWidget(button)
        return row

    def _browse_material(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择素材文件夹（语音 / 视频）")
        if path:
            self.material_input.setText(path)

    def _browse_vocal(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择歌曲", "", "Audio (*.wav *.flac *.mp3 *.m4a *.ogg);;All files (*)"
        )
        if path:
            self.vocal_input.setText(path)

    def _browse_lyrics(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择歌词", "", "Lyrics (*.txt *.lrc)")
        if path:
            self.lyrics_input.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_input.setText(path)

    # ------------------------------------------------------------- running

    def _start(self) -> None:
        material, vocal = self.material_input.text().strip(), self.vocal_input.text().strip()
        if not material or not Path(material).exists():
            QMessageBox.warning(self, "一键人力", "请选择素材文件夹。")
            return
        if not vocal or not Path(vocal).exists():
            QMessageBox.warning(self, "一键人力", "请选择歌曲文件。")
            return
        options = MakeOptions(
            material=[material],
            vocal=vocal,
            output=self.output_input.text().strip(),
            lyrics=self.lyrics_input.text().strip() or None,
            name=self.name_input.text().strip() or "HumanSlice",
            separate=self.separate_check.isChecked(),
            separate_material=self.separate_material_check.isChecked(),
            render=RenderOptions(
                select=SelectOptions(join_weight=self.join_spin.value()),
                backend=self.backend_combo.currentData(),
            ),
        )
        self.log.clear()
        self.run_button.setEnabled(False)
        self._thread = QThread(self)
        worker = MakeWorker(options)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self.log.appendPlainText)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        for signal in (worker.finished, worker.failed):
            signal.connect(self._thread.quit)
            signal.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker = worker  # keep a reference while running
        self._thread.start()

    def _on_finished(self, results: dict) -> None:
        self._results = results
        self.run_button.setEnabled(True)
        self.open_mix_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)
        self.log.appendPlainText("\n输出：")
        for key, path in results.items():
            self.log.appendPlainText(f"  {key}: {path}")

    def _on_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "一键人力", f"制作失败：{message}")

    def _open_result(self) -> None:
        self._open_path(self._results.get("mix") or self._results.get("vocal", ""))

    @staticmethod
    def _open_path(path: str) -> None:
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "一键人力", "制作仍在进行，请等待完成后再关闭。")
            event.ignore()
            return
        super().closeEvent(event)
