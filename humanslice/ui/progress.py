from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget

# The analysis layer reports progress in English; the UI shows Chinese.
_PROGRESS_MESSAGES_ZH = {
    "Preparing audio for segmentation...": "正在准备音频...",
    "Computing onset and energy features...": "正在计算起音、能量与频谱特征...",
    "Combining segmentation cues...": "正在融合切分线索...",
    "Detecting onset and silence boundaries...": "正在检测起音与无声边界...",
    "Refining weak boundaries and breath fragments...": "正在合并弱边界与呼吸音碎片...",
    "Candidate cutpoints ready.": "候选切点已生成，正在分析片段...",
}


def create_progress_dialog(parent: QWidget, title: str, label: str) -> QProgressDialog:
    """Show a modal, non-cancellable 0-100 progress dialog immediately."""
    dialog = QProgressDialog(label, "", 0, 100, parent)
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


def update_progress(dialog: QProgressDialog, value: int, message: str | None = None) -> None:
    if message:
        dialog.setLabelText(message)
    dialog.setValue(max(0, min(100, value)))
    QApplication.processEvents()


def translate_progress_message(message: str) -> str:
    return _PROGRESS_MESSAGES_ZH.get(message, message)
