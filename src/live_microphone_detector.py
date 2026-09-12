import argparse
import warnings
from collections import deque
from queue import Empty, Queue
from typing import Deque, Dict, List

import numpy as np

from audio_preprocessing import TARGET_SAMPLE_RATE, preprocess_samples
from voiceshield_inference import VoiceShieldModel


warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
)

try:
    import sounddevice as sd
except ImportError:
    sd = None


CHUNK_SECONDS = 2.0
ROLLING_WINDOW_CHUNKS = 5


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def risk_level_from_fake_score(fake_score: float, thresholds: Dict[str, object]) -> str:
    real_threshold = float(thresholds["real_if_fake_score_at_or_below"])
    fake_threshold = float(thresholds["fake_if_fake_score_at_or_above"])

    if fake_score >= fake_threshold:
        return "HIGH RISK"
    if fake_score <= real_threshold:
        return "SAFE"
    return "SUSPICIOUS"


def rolling_call_risk(recent_chunks: Deque[Dict[str, object]], thresholds: Dict[str, object]) -> Dict[str, object]:
    if not recent_chunks:
        return {
            "risk_level": "NO AUDIO",
            "rolling_risk_score": 0.0,
            "average_fake_score": 0.0,
            "maximum_fake_score": 0.0,
            "high_risk_chunks": 0,
            "suspicious_chunks": 0,
        }

    fake_scores = np.asarray([float(chunk["fake_score"]) for chunk in recent_chunks], dtype=np.float32)
    levels = [str(chunk["risk_level"]) for chunk in recent_chunks]
    high_risk_count = sum(level == "HIGH RISK" for level in levels)
    suspicious_count = sum(level == "SUSPICIOUS" for level in levels)
    average_fake_score = float(fake_scores.mean())
    maximum_fake_score = float(fake_scores.max())
    high_risk_percent = high_risk_count / len(recent_chunks)

    rolling_risk_score = float(
        0.60 * average_fake_score
        + 0.25 * maximum_fake_score
        + 0.15 * high_risk_percent
    )

    real_threshold = float(thresholds["real_if_fake_score_at_or_below"])
    fake_threshold = float(thresholds["fake_if_fake_score_at_or_above"])

    has_enough_context = len(recent_chunks) >= 3

    if high_risk_count >= 2 or (
        has_enough_context
        and high_risk_percent >= 0.40
        and average_fake_score >= fake_threshold
    ):
        risk_level = "HIGH RISK"
    elif high_risk_count == 1 or suspicious_count > 0 or average_fake_score > real_threshold:
        risk_level = "SUSPICIOUS"
    else:
        risk_level = "SAFE"

    return {
        "risk_level": risk_level,
        "rolling_risk_score": rolling_risk_score,
        "average_fake_score": average_fake_score,
        "maximum_fake_score": maximum_fake_score,
        "high_risk_chunks": high_risk_count,
        "suspicious_chunks": suspicious_count,
    }


def analyze_microphone_stream(
    chunk_seconds: float = CHUNK_SECONDS,
    rolling_window_chunks: int = ROLLING_WINDOW_CHUNKS,
) -> List[Dict[str, object]]:
    if sd is None:
        raise RuntimeError("sounddevice is not installed. Run: python -m pip install sounddevice")

    model = VoiceShieldModel()
    thresholds = model.metadata["decision_thresholds"]
    chunk_size = int(round(chunk_seconds * TARGET_SAMPLE_RATE))
    audio_queue: Queue[np.ndarray] = Queue()
    recent_chunks: Deque[Dict[str, object]] = deque(maxlen=rolling_window_chunks)
    all_chunks: List[Dict[str, object]] = []

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print("Microphone status:", status)
        audio_queue.put(indata.copy())

    print("VOICE SHIELD LIVE MICROPHONE DETECTION")
    print("--------------------------------------")
    print("Press Ctrl+C to stop.")
    print("Sample rate:", TARGET_SAMPLE_RATE)
    print("Chunk length:", f"{chunk_seconds:.1f} sec")
    print("Rolling window:", rolling_window_chunks, "chunks")
    print("Decision thresholds:", thresholds)

    try:
        with sd.InputStream(
            samplerate=TARGET_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback,
        ):
            chunk_number = 0
            while True:
                try:
                    raw_chunk = audio_queue.get(timeout=1.0)
                except Empty:
                    continue

                audio = preprocess_samples(raw_chunk, TARGET_SAMPLE_RATE)
                if not audio.valid:
                    print(f"\nChunk skipped: {audio.warning}")
                    continue

                chunk_number += 1
                result = model.predict_samples(
                    audio.samples,
                    audio.sample_rate,
                    audio_info={
                        "duration_seconds": audio.duration_seconds,
                        "rms": audio.rms,
                        "was_trimmed": audio.was_trimmed,
                    },
                )
                risk_level = risk_level_from_fake_score(result["fake_score"], thresholds)
                chunk_result = {
                    "chunk": chunk_number,
                    "real_score": float(result["real_score"]),
                    "fake_score": float(result["fake_score"]),
                    "risk_score": float(result["risk_score"]),
                    "risk_level": risk_level,
                    "binary_prediction": result["binary_prediction"],
                    "model_verdict": result["verdict"],
                }
                all_chunks.append(chunk_result)
                recent_chunks.append(chunk_result)
                call_risk = rolling_call_risk(recent_chunks, thresholds)

                print(f"\nChunk {chunk_number}")
                print("REAL:", percent(chunk_result["real_score"]))
                print("FAKE:", percent(chunk_result["fake_score"]))
                print("Status:", chunk_result["risk_level"])
                print("Rolling fake score:", percent(call_risk["rolling_risk_score"]))
                print("Current Call Risk:", call_risk["risk_level"])

    except KeyboardInterrupt:
        print("\nStopped microphone detection.")
    except Exception as exc:
        raise RuntimeError(f"Microphone detection failed: {exc}") from exc

    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live microphone VoiceShield detection.")
    parser.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    parser.add_argument("--rolling-window", type=int, default=ROLLING_WINDOW_CHUNKS)
    args = parser.parse_args()

    analyze_microphone_stream(
        chunk_seconds=args.chunk_seconds,
        rolling_window_chunks=args.rolling_window,
    )


if __name__ == "__main__":
    main()
