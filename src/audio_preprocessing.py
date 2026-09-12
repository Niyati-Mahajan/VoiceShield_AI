from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np


TARGET_SAMPLE_RATE = 16000
PREPROCESSING_VERSION = "audio-preprocess-v2"


@dataclass
class AudioConfig:
    sample_rate: int = TARGET_SAMPLE_RATE
    trim_top_db: float = 35.0
    min_duration_seconds: float = 0.35
    near_silent_rms: float = 1e-4
    peak_normalize_to: float = 0.95


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    rms: float
    peak: float
    was_trimmed: bool
    valid: bool
    warning: Optional[str] = None


def preprocess_samples(
    samples: np.ndarray,
    input_sample_rate: int,
    config: AudioConfig | None = None,
) -> AudioData:
    """Apply the shared VoiceShield preprocessing to already-loaded audio samples."""
    config = config or AudioConfig()
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    if input_sample_rate != config.sample_rate and samples.size > 0:
        samples = librosa.resample(
            samples,
            orig_sr=input_sample_rate,
            target_sr=config.sample_rate,
        ).astype(np.float32, copy=False)

    if samples.size == 0:
        return AudioData(
            samples=samples,
            sample_rate=config.sample_rate,
            duration_seconds=0.0,
            rms=0.0,
            peak=0.0,
            was_trimmed=False,
            valid=False,
            warning="empty audio",
        )

    original_length = len(samples)
    trimmed, index = librosa.effects.trim(
        samples,
        top_db=config.trim_top_db,
        frame_length=2048,
        hop_length=512,
    )
    if trimmed.size > 0:
        samples = trimmed.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1e-8:
        samples = (samples / peak * config.peak_normalize_to).astype(np.float32)

    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    duration = float(samples.size / config.sample_rate)
    warning = None
    valid = True

    if duration < config.min_duration_seconds:
        valid = False
        warning = f"audio too short after preprocessing ({duration:.2f}s)"
    elif rms < config.near_silent_rms:
        valid = False
        warning = f"audio is near silent after preprocessing (rms={rms:.6f})"

    return AudioData(
        samples=samples,
        sample_rate=config.sample_rate,
        duration_seconds=duration,
        rms=rms,
        peak=peak,
        was_trimmed=bool(index[0] > 0 or index[1] < original_length),
        valid=valid,
        warning=warning,
    )


def load_audio(path: str | Path, config: AudioConfig | None = None) -> AudioData:
    """Load audio exactly once for both training and inference."""
    config = config or AudioConfig()
    samples, sr = librosa.load(str(path), sr=None, mono=True)
    return preprocess_samples(samples, sr, config)


def augment_audio(samples: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Conservative training-only augmentation for microphone/phone variation."""
    augmented = np.asarray(samples, dtype=np.float32).copy()

    if rng.random() < 0.70:
        gain = float(rng.uniform(0.80, 1.15))
        augmented *= gain

    if rng.random() < 0.45:
        rms = float(np.sqrt(np.mean(np.square(augmented)))) if augmented.size else 0.0
        noise_level = rms * float(rng.uniform(0.002, 0.010))
        augmented += rng.normal(0.0, noise_level, size=augmented.shape).astype(np.float32)

    if rng.random() < 0.25 and augmented.size > sample_rate:
        degraded_sr = int(rng.choice([8000, 11025, 12000]))
        degraded = librosa.resample(augmented, orig_sr=sample_rate, target_sr=degraded_sr)
        augmented = librosa.resample(degraded, orig_sr=degraded_sr, target_sr=sample_rate)

    peak = float(np.max(np.abs(augmented))) if augmented.size else 0.0
    if peak > 1.0:
        augmented = augmented / peak

    return augmented.astype(np.float32, copy=False)


def audio_config_dict(config: AudioConfig | None = None) -> Dict[str, float | int | str]:
    config = config or AudioConfig()
    return {
        "version": PREPROCESSING_VERSION,
        "sample_rate": config.sample_rate,
        "trim_top_db": config.trim_top_db,
        "min_duration_seconds": config.min_duration_seconds,
        "near_silent_rms": config.near_silent_rms,
        "peak_normalize_to": config.peak_normalize_to,
    }
