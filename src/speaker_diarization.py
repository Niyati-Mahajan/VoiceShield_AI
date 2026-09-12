import argparse
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import torch


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")


class DiarizationSetupError(RuntimeError):
    """Raised when pyannote diarization is not available or not configured."""


def _hf_token() -> Optional[str]:
    for name in HF_TOKEN_ENV_VARS:
        token = os.getenv(name)
        if token:
            return token
    return None


@lru_cache(maxsize=1)
def load_diarization_pipeline(
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    device: Optional[str] = None,
):
    """Load the pyannote pipeline once per process."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationSetupError(
            "pyannote.audio is not installed. Install it with: pip install pyannote.audio"
        ) from exc

    token = _hf_token()
    if not token:
        raise DiarizationSetupError(
            "Hugging Face token not found. Set HF_TOKEN after accepting access to "
            f"{model_name} on Hugging Face."
        )

    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
    except Exception as exc:
        raise DiarizationSetupError(
            "Could not load the pyannote diarization model. Make sure your Hugging Face "
            f"account has accepted the gated model conditions for {model_name}, and "
            "that HF_TOKEN is set to a token with read access."
        ) from exc

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(pipeline, "to"):
        pipeline.to(torch.device(target_device))
    return pipeline


def diarize_audio(
    audio_path: str | Path,
    *,
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    device: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    Return speaker turns for an audio file.

    Each item has: speaker, start, end. If the model reports overlapping speech,
    overlapping time ranges are preserved as separate items.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    pipeline = load_diarization_pipeline(model_name=model_name, device=device)
    diarization = pipeline(str(path))

    segments: List[Dict[str, object]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end <= start:
            continue
        segments.append(
            {
                "speaker": str(speaker),
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )

    return sorted(segments, key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])))


def print_diarization(segments: List[Dict[str, object]]) -> None:
    if not segments:
        print("No speech segments detected.")
        return

    for segment in segments:
        print(f"{segment['speaker']}: {segment['start']:.2f}s - {segment['end']:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run speaker diarization on an audio file.")
    parser.add_argument("audio_file")
    parser.add_argument("--model", default=DEFAULT_DIARIZATION_MODEL)
    args = parser.parse_args()

    segments = diarize_audio(args.audio_file, model_name=args.model)
    print_diarization(segments)


if __name__ == "__main__":
    main()
