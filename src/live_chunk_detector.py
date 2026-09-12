import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from audio_preprocessing import load_audio
from voiceshield_inference import VoiceShieldModel


CHUNK_SECONDS = 2.0
MIN_FINAL_CHUNK_SECONDS = 0.75


def risk_level_from_fake_score(fake_score: float, thresholds: Dict[str, object]) -> str:
    real_threshold = float(thresholds["real_if_fake_score_at_or_below"])
    fake_threshold = float(thresholds["fake_if_fake_score_at_or_above"])

    if fake_score >= fake_threshold:
        return "HIGH RISK"
    if fake_score <= real_threshold:
        return "SAFE"
    return "SUSPICIOUS"


def split_into_chunks(
    samples: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = CHUNK_SECONDS,
    min_final_chunk_seconds: float = MIN_FINAL_CHUNK_SECONDS,
) -> tuple[List[Dict[str, object]], List[str]]:
    chunk_size = int(round(chunk_seconds * sample_rate))
    min_final_size = int(round(min_final_chunk_seconds * sample_rate))
    chunks: List[Dict[str, object]] = []
    messages: List[str] = []

    start = 0
    while start < len(samples):
        end = min(start + chunk_size, len(samples))
        chunk_samples = samples[start:end]

        if len(chunk_samples) < min_final_size and chunks:
            previous = chunks[-1]
            previous_samples = previous["samples"]
            previous["samples"] = np.concatenate([previous_samples, chunk_samples])
            previous["end"] = len(samples) / sample_rate
            messages.append(
                f"Final short chunk ({len(chunk_samples) / sample_rate:.2f}s) merged with previous chunk."
            )
            break

        if len(chunk_samples) < min_final_size:
            messages.append(
                f"Skipped final short chunk ({len(chunk_samples) / sample_rate:.2f}s)."
            )
            break

        chunks.append(
            {
                "start": start / sample_rate,
                "end": end / sample_rate,
                "samples": chunk_samples,
            }
        )
        start = end

    return chunks, messages


def aggregate_chunk_results(chunks: List[Dict[str, object]], thresholds: Dict[str, object]) -> Dict[str, object]:
    if not chunks:
        return {
            "overall_risk_score": 0.0,
            "risk_level": "NO USABLE AUDIO",
            "average_fake_score": 0.0,
            "maximum_fake_score": 0.0,
            "suspicious_chunks": 0,
            "high_risk_chunks": 0,
            "suspicious_chunk_percent": 0.0,
            "high_risk_chunk_percent": 0.0,
        }

    fake_scores = np.asarray([float(chunk["fake_score"]) for chunk in chunks], dtype=np.float32)
    risk_levels = [str(chunk["risk_level"]) for chunk in chunks]
    suspicious_count = sum(level == "SUSPICIOUS" for level in risk_levels)
    high_risk_count = sum(level == "HIGH RISK" for level in risk_levels)
    suspicious_percent = suspicious_count / len(chunks)
    high_risk_percent = high_risk_count / len(chunks)
    average_fake_score = float(fake_scores.mean())
    maximum_fake_score = float(fake_scores.max())

    overall_risk_score = float(
        0.55 * average_fake_score
        + 0.30 * maximum_fake_score
        + 0.15 * high_risk_percent
    )

    real_threshold = float(thresholds["real_if_fake_score_at_or_below"])
    fake_threshold = float(thresholds["fake_if_fake_score_at_or_above"])

    if high_risk_count >= 2 or high_risk_percent >= 0.25 or average_fake_score >= fake_threshold:
        risk_level = "HIGH RISK"
    elif high_risk_count == 1 or suspicious_percent >= 0.25 or average_fake_score > real_threshold:
        risk_level = "SUSPICIOUS"
    else:
        risk_level = "SAFE"

    return {
        "overall_risk_score": overall_risk_score,
        "risk_level": risk_level,
        "average_fake_score": average_fake_score,
        "maximum_fake_score": maximum_fake_score,
        "suspicious_chunks": suspicious_count,
        "high_risk_chunks": high_risk_count,
        "suspicious_chunk_percent": suspicious_percent,
        "high_risk_chunk_percent": high_risk_percent,
    }


def analyze_audio_file(audio_path: str | Path) -> Dict[str, object]:
    audio = load_audio(audio_path)
    if not audio.valid:
        raise ValueError(audio.warning or "audio is not usable")

    model = VoiceShieldModel()
    raw_chunks, messages = split_into_chunks(audio.samples, audio.sample_rate)
    analyzed_chunks: List[Dict[str, object]] = []

    for chunk in raw_chunks:
        samples = chunk["samples"]
        result = model.predict_samples(
            samples,
            audio.sample_rate,
            audio_info={
                "duration_seconds": len(samples) / audio.sample_rate,
                "start": chunk["start"],
                "end": chunk["end"],
            },
        )
        risk_level = risk_level_from_fake_score(result["fake_score"], result["thresholds"])
        analyzed_chunks.append(
            {
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "real_score": float(result["real_score"]),
                "fake_score": float(result["fake_score"]),
                "risk_score": float(result["risk_score"]),
                "risk_level": risk_level,
                "binary_prediction": result["binary_prediction"],
                "model_verdict": result["verdict"],
            }
        )

    thresholds = model.metadata["decision_thresholds"]
    summary = aggregate_chunk_results(analyzed_chunks, thresholds)
    return {
        "audio_path": str(audio_path),
        "total_duration": audio.duration_seconds,
        "sample_rate": audio.sample_rate,
        "chunk_seconds": CHUNK_SECONDS,
        "min_final_chunk_seconds": MIN_FINAL_CHUNK_SECONDS,
        "thresholds": thresholds,
        "messages": messages,
        "overall_risk_score": summary["overall_risk_score"],
        "risk_level": summary["risk_level"],
        "average_fake_score": summary["average_fake_score"],
        "maximum_fake_score": summary["maximum_fake_score"],
        "suspicious_chunks": summary["suspicious_chunks"],
        "high_risk_chunks": summary["high_risk_chunks"],
        "suspicious_chunk_percent": summary["suspicious_chunk_percent"],
        "high_risk_chunk_percent": summary["high_risk_chunk_percent"],
        "chunks": analyzed_chunks,
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_analysis(analysis: Dict[str, object]) -> None:
    print("\nVOICE SHIELD CHUNK ANALYSIS")
    print("---------------------------")
    print("File:", analysis["audio_path"])
    print("Decision thresholds:", analysis["thresholds"])

    for message in analysis["messages"]:
        print("Note:", message)

    chunks = analysis["chunks"]
    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index} | {chunk['start']:.1f}-{chunk['end']:.1f} sec")
        print("REAL:", percent(chunk["real_score"]))
        print("FAKE:", percent(chunk["fake_score"]))
        print("Verdict:", chunk["risk_level"])

    print("\n## VOICE SHIELD LIVE ANALYSIS")
    print("Total duration:", f"{analysis['total_duration']:.2f} sec")
    print("Chunks analyzed:", len(chunks))
    print("Average fake score:", percent(analysis["average_fake_score"]))
    print("Maximum fake score:", percent(analysis["maximum_fake_score"]))
    print(
        "Suspicious chunks:",
        f"{analysis['suspicious_chunks']} ({percent(analysis['suspicious_chunk_percent'])})",
    )
    print(
        "High-risk chunks:",
        f"{analysis['high_risk_chunks']} ({percent(analysis['high_risk_chunk_percent'])})",
    )
    print("Final risk score:", percent(analysis["overall_risk_score"]))
    print("Final verdict:", analysis["risk_level"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunk-based VoiceShield detection on one audio file.")
    parser.add_argument("audio_file")
    args = parser.parse_args()

    analysis = analyze_audio_file(args.audio_file)
    print_analysis(analysis)


if __name__ == "__main__":
    main()
