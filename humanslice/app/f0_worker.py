from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from humanslice.analysis.scoring import ScoreConfig, compute_preview_f0_curve


class F0PreviewWorker(QObject):
    """Background worker for whole-track F0 preview generation."""

    finished = Signal(int, object, object)
    failed = Signal(int, str)

    def __init__(
        self,
        job_id: int,
        samples: np.ndarray,
        sample_rate: int,
        config: ScoreConfig,
    ) -> None:
        super().__init__()
        self._job_id = job_id
        self._samples = samples
        self._sample_rate = sample_rate
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            times, values = compute_preview_f0_curve(self._samples, self._sample_rate, self._config)
        except Exception as exc:
            self.failed.emit(self._job_id, str(exc))
            return
        self.finished.emit(self._job_id, times, values)
