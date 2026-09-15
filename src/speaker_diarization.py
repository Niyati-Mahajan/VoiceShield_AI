import argparse
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import torch


DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
PYANNOTE_ACCESS_URLS = (
    "https://huggingface.co/pyannote/speaker-diarization-community-1",
    "https://huggingface.co/pyannote/segmentation-community-1",
)


class DiarizationSetupError(RuntimeError):
    """Raised when pyannote diarization is unavailable or not configured."""


class DiarizationRuntimeError(RuntimeError):
    """Raised when pyannote cannot diarize the requested audio."""


def _hf_token() -> Optional[str]:
    """Read Hugging Face credentials from the environment or local HF cache."""
    for name in HF_TOKEN_ENV_VARS:
        token = os.getenv(name)
        if token:
            return token

    try:
        from huggingface_hub import get_token
    except ImportError:
        return None

    return get_token()


@lru_cache(maxsize=1)
def load_diarization_pipeline(
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    device: Optional[str] = None,
):
    """Load the pyannote diarization pipeline once per Python process."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationSetupError(
            "pyannote.audio is not installed. Install it with: pip install pyannote.audio"
        ) from exc

    token = _hf_token()
    try:
        if token:
            try:
                pipeline = Pipeline.from_pretrained(model_name, token=token)
            except TypeError:
                pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        else:
            pipeline = Pipeline.from_pretrained(model_name)
    except Exception as exc:
        raise DiarizationSetupError(
            "Could not load the pyannote diarization model. Make sure your Hugging Face "
            "account has accepted the required pyannote model terms, and that a read "
            "token is available through HF_TOKEN or `huggingface-cli login`. Required "
            f"repositories: {', '.join(PYANNOTE_ACCESS_URLS)}"
        ) from exc

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(pipeline, "to"):
        pipeline.to(torch.device(target_device))
    return pipeline


def _extract_annotation(output):
    """Support pyannote 4 output wrappers and older Annotation returns."""
    if hasattr(output, "speaker_diarization"):
        return output.speaker_diarization

    for attribute in ("diarization", "annotation"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value

    if isinstance(output, dict):
        for key in ("speaker_diarization", "diarization", "annotation"):
            value = output.get(key)
            if value is not None:
                return value

    return output


def _iter_speaker_turns(output) -> Iterable[Tuple[object, str]]:
    annotation = _extract_annotation(output)

    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            yield turn, str(speaker)
        return

    try:
        iterator = iter(annotation)
    except TypeError as exc:
        raise TypeError("Unsupported pyannote diarization output format.") from exc

    for item in iterator:
        if isinstance(item, tuple) and len(item) == 2:
            turn, speaker = item
            yield turn, str(speaker)
        elif isinstance(item, tuple) and len(item) >= 3:
            turn, _, speaker = item[:3]
            yield turn, str(speaker)
        else:
            raise TypeError("Unsupported pyannote diarization turn format.")


def _speaker_count(segments: List[Dict[str, object]]) -> int:
    return len({str(segment["speaker"]) for segment in segments})


def _load_waveform(audio_path: Path) -> Dict[str, object]:
    """Decode audio with librosa so pyannote does not require torchcodec."""
    try:
        samples, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not load audio file {audio_path}: {exc}") from exc

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")

    waveform = torch.from_numpy(samples).float().unsqueeze(0)
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def diarize_audio(
    audio_path: str | Path,
    *,
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    Return structured speaker diarization data for one audio file.

    The detector/classifier pipeline is intentionally not used here. This module
    only answers: who spoke when?
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")

    pyannote_input = _load_waveform(path)
    pipeline = load_diarization_pipeline(model_name=model_name, device=device)

    try:
        output = pipeline(pyannote_input)
    except Exception as exc:
        raise DiarizationRuntimeError(f"Diarization failed for {path}: {exc}") from exc

    segments: List[Dict[str, object]] = []
    try:
        for turn, speaker in _iter_speaker_turns(output):
            start = float(turn.start)
            end = float(turn.end)
            if end <= start:
                continue
            segments.append(
                {
                    "speaker": speaker,
                    "start": round(start, 2),
                    "end": round(end, 2),
                }
            )
    except Exception as exc:
        raise DiarizationRuntimeError(f"Could not parse diarization output: {exc}") from exc

    segments = sorted(
        segments,
        key=lambda item: (str(item["speaker"]), float(item["start"]), float(item["end"])),
    )
    return {
        "num_speakers": _speaker_count(segments),
        "segments": segments,
    }


def print_diarization(result: Dict[str, object]) -> None:
    segments = list(result.get("segments", []))
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for segment in segments:
        grouped[str(segment["speaker"])].append(segment)

    print("## VOICE SHIELD SPEAKER DIARIZATION")
    print()
    print(f"Detected speakers: {int(result.get('num_speakers', 0))}")

    if not segments:
        print()
        print("No speech segments detected.")
        return

    for speaker in sorted(grouped):
        print()
        print(speaker)
        for segment in grouped[speaker]:
            print(f"{float(segment['start']):.2f} - {float(segment['end']):.2f} sec")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VoiceShield speaker diarization on one audio file.")
    parser.add_argument("audio_file")
    parser.add_argument("--model", default=DEFAULT_DIARIZATION_MODEL)
    args = parser.parse_args()

    try:
        result = diarize_audio(args.audio_file, model_name=args.model)
    except (DiarizationSetupError, DiarizationRuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Speaker diarization error: {exc}", file=sys.stderr)
        return 1

    print_diarization(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
