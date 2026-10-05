"""Speech recognition and character-level timestamps with Qwen3-ASR / Qwen3-ForcedAligner.

Both run through native Hugging Face Transformers (>= 5.13). On an 8 GB GPU load
one model at a time and call ``close()`` before loading the next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
SAMPLE_RATE = 16000
_LANGUAGE_NAMES = {"zh": "Chinese", "ja": "Japanese", "en": "English", "yue": "Cantonese"}


@dataclass(slots=True)
class TimedToken:
    text: str
    start: float
    end: float


class QwenASR:
    def __init__(self, model_id: str = ASR_MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        from humanslice.common.models import torch_device

        self.device = torch_device(device)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()

    def transcribe(
        self,
        audios: list[np.ndarray],
        language: str | None = "zh",
        batch_size: int = 8,
        max_new_tokens: int = 384,
    ) -> list[str]:
        """Transcribe 16 kHz mono clips (each up to ~30 s)."""
        import torch

        texts: list[str] = []
        for offset in range(0, len(audios), batch_size):
            batch = [np.ascontiguousarray(audio, dtype=np.float32) for audio in audios[offset:offset + batch_size]]
            inputs = self.processor.apply_transcription_request(
                batch,
                language=[language] * len(batch) if language else None,
                processor_kwargs={"sampling_rate": SAMPLE_RATE},
            ).to(self.model.device, self.model.dtype)
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            texts.extend(
                text.strip()
                for text in self.processor.decode(generated, return_format="transcription_only")
            )
        return texts

    def close(self) -> None:
        from humanslice.common.models import release_gpu_memory

        del self.model
        release_gpu_memory()


class QwenAligner:
    """Character-level timestamps for Chinese / Japanese (word-level for spaced languages)."""

    def __init__(self, model_id: str = ALIGNER_MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        from humanslice.common.models import torch_device

        self.device = torch_device(device)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id, dtype=dtype).to(self.device).eval()

    def align(self, audio: np.ndarray, text: str, language: str = "zh") -> list[TimedToken]:
        """Align ``text`` to 16 kHz ``audio`` (<= 5 minutes)."""
        import torch

        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=np.ascontiguousarray(audio, dtype=np.float32),
            transcript=text,
            language=_LANGUAGE_NAMES.get(language, language),
            processor_kwargs={"sampling_rate": SAMPLE_RATE},
        )
        inputs = inputs.to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        stamps = self.processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.model.config.timestamp_token_id,
        )[0]
        return [TimedToken(item["text"], float(item["start_time"]), float(item["end_time"])) for item in stamps]

    def close(self) -> None:
        from humanslice.common.models import release_gpu_memory

        del self.model
        release_gpu_memory()
