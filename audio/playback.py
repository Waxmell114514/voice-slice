from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from audio.io import extract_clip
from models.project import AudioTrack, Segment


class PlaybackController(QObject):
    """Preview audio by rendering temporary wav files for QMediaPlayer."""

    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, volume: int = 85, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(max(0.0, min(1.0, volume / 100.0)))
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.errorOccurred.connect(self._on_error)
        self._temp_dir = Path(tempfile.mkdtemp(prefix="humanslice_preview_"))

    def play_full_track(self, track: AudioTrack) -> None:
        """Play the full loaded track."""
        self.play_array(track.samples, track.sample_rate)

    def play_segment(self, track: AudioTrack, segment: Segment) -> None:
        """Play a single segment from the loaded track."""
        self.play_array(extract_clip(track.samples, track.sample_rate, segment.start, segment.end), track.sample_rate)

    def play_boundary_window(self, track: AudioTrack, boundary_time: float, window_ms: int = 180) -> None:
        """Play a short window centered around a boundary time."""
        half_window = max(0.02, window_ms / 2000.0)
        start = max(0.0, boundary_time - half_window)
        end = min(track.duration, boundary_time + half_window)
        self.play_array(extract_clip(track.samples, track.sample_rate, start, end), track.sample_rate)

    def play_transition_preview(
        self,
        track: AudioTrack,
        left_segment: Segment,
        right_segment: Segment,
        crossfade_ms: int = 35,
    ) -> None:
        """Play a quick stitched preview of two adjacent segments."""
        left_clip = extract_clip(
            track.samples,
            track.sample_rate,
            max(left_segment.start, left_segment.end - 0.2),
            left_segment.end,
        )
        right_clip = extract_clip(
            track.samples,
            track.sample_rate,
            right_segment.start,
            min(right_segment.end, right_segment.start + 0.2),
        )
        if left_clip.size == 0 or right_clip.size == 0:
            self.error_occurred.emit("无法生成相邻片段拼接试听。")
            return
        crossfade_samples = max(8, int(track.sample_rate * max(0, crossfade_ms) / 1000.0))
        self.play_array(self._crossfade(left_clip, right_clip, crossfade_samples), track.sample_rate)

    def play_array(self, samples: np.ndarray, sample_rate: int) -> None:
        """Render a numpy buffer to temp wav and play it."""
        if samples.size == 0:
            self.error_occurred.emit("没有可播放的音频内容。")
            return
        wav_path = self._write_temp_wav(samples, sample_rate)
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(wav_path)))
        self._player.play()
        self.status_changed.emit(f"试听中: {wav_path.name}")

    def stop(self) -> None:
        """Stop current playback."""
        self._player.stop()

    def cleanup(self) -> None:
        """Delete temporary preview files when the app exits."""
        self._player.stop()
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _write_temp_wav(self, samples: np.ndarray, sample_rate: int) -> Path:
        clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
        file_path = self._temp_dir / f"preview_{uuid.uuid4().hex}.wav"
        sf.write(file_path, clipped, sample_rate, subtype="PCM_16")
        return file_path

    def _crossfade(self, left: np.ndarray, right: np.ndarray, crossfade_samples: int) -> np.ndarray:
        crossfade_samples = min(crossfade_samples, left.size, right.size)
        if crossfade_samples <= 0:
            return np.concatenate([left, right]).astype(np.float32, copy=False)

        fade_out = np.linspace(1.0, 0.0, num=crossfade_samples, endpoint=False, dtype=np.float32)
        fade_in = 1.0 - fade_out
        overlap = left[-crossfade_samples:] * fade_out + right[:crossfade_samples] * fade_in
        stitched = np.concatenate([left[:-crossfade_samples], overlap, right[crossfade_samples:]])
        return np.ascontiguousarray(stitched, dtype=np.float32)

    def _on_error(self, error: QMediaPlayer.Error, error_text: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        self.error_occurred.emit(error_text or "播放器发生未知错误。")
