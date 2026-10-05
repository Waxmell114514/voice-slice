from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QWidget

from humanslice.models.project import RegionRange, Segment


class SegmentDetailsPanel(QGroupBox):
    """Read-only summary of the current segment's bounds, scores, and regions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("当前片段", parent)
        form = QFormLayout(self)
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
        form.addRow("ID", self.segment_id_label)
        form.addRow("开始", self.start_label)
        form.addRow("结束", self.end_label)
        form.addRow("时长", self.duration_label)
        form.addRow("Clarity", self.clarity_label)
        form.addRow("Stability", self.stability_label)
        form.addRow("推荐角色", self.role_label)
        form.addRow("Onset", self.onset_label)
        form.addRow("Nucleus", self.nucleus_label)
        form.addRow("Tail", self.tail_label)
        self._value_labels = (
            self.segment_id_label,
            self.start_label,
            self.end_label,
            self.duration_label,
            self.clarity_label,
            self.stability_label,
            self.role_label,
            self.onset_label,
            self.nucleus_label,
            self.tail_label,
        )

    def show_segment(self, segment: Segment | None) -> None:
        if segment is None:
            for label in self._value_labels:
                label.setText("-")
            return

        self.segment_id_label.setText(segment.segment_id)
        self.start_label.setText(f"{segment.start:.4f} s")
        self.end_label.setText(f"{segment.end:.4f} s")
        self.duration_label.setText(f"{segment.duration:.4f} s")

        score = segment.score
        self.clarity_label.setText("-" if score is None else f"{score.clarity_score:.1f}")
        self.stability_label.setText("-" if score is None else f"{score.stability_score:.1f}")
        self.role_label.setText("-" if score is None else score.recommended_role)

        regions = segment.regions
        self.onset_label.setText(_format_region(regions.onset if regions else None))
        self.nucleus_label.setText(_format_region(regions.nucleus if regions else None))
        self.tail_label.setText(_format_region(regions.tail if regions else None))


def _format_region(region: RegionRange | None) -> str:
    if region is None:
        return "-"
    return f"{region.start:.4f} - {region.end:.4f} s"
