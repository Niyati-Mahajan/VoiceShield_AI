import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np

from audio_preprocessing import AudioConfig, TARGET_SAMPLE_RATE, load_audio, preprocess_samples
from live_chunk_detector import analyze_audio_file as analyze_chunked_audio_file
from live_chunk_detector import risk_level_from_fake_score
from speaker_diarization import DiarizationSetupError, diarize_audio
from voiceshield_inference import VoiceShieldModel


MIN_SPEECH_SECONDS = 1.0
MIN_RELIABLE_SPEAKER_SECONDS = 1.0
STRONG_FAKE_SCORE = 0.85
MIN_HIGH_RISK_SEGMENTS_FOR_CALL = 1


def _empty_speaker_result(speaker_id: str, segments: List[Dict[str, object]], reason: str) -> Dict[str, object]:
    total_duration = sum(float(segment["end"]) - float(segment["start"]) for segment in segments)
    return {
        "speaker_id": speaker_id,
        "speech_duration": round(total_duration, 3),
        "segment_count": len(segments),
        "reliable_segment_count": 0,
        "real_score": None,
        "fake_score": None,
        "risk_score": None,
        "risk_level": "INSUFFICIENT_AUDIO",
        "status": "INSUFFICIENT_AUDIO",
        "reason": reason,
        "segments": segments,
    }


def _slice_segment(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_index = max(0, int(round(start * sample_rate)))
    end_index = min(len(samples), int(round(end * sample_rate)))
    if end_index <= start_index:
        return np.asarray([], dtype=np.float32)
    return samples[start_index:end_index]


def _analyze_speaker_segments(
    speaker_id: str,
    segments: List[Dict[str, object]],
    samples: np.ndarray,
    sample_rate: int,
    model: VoiceShieldModel,
    audio_config: Optional[AudioConfig] = None,
    min_speech_seconds: float = MIN_SPEECH_SECONDS,
) -> Dict[str, object]:
    audio_config = audio_config or AudioConfig()
    analyzed_segments: List[Dict[str, object]] = []
    reliable_results: List[Dict[str, object]] = []
    total_speech_duration = 0.0

    for segment in segments:
        raw = _slice_segment(samples, sample_rate, float(segment["start"]), float(segment["end"]))
        raw_duration = float(len(raw) / sample_rate) if sample_rate else 0.0
        total_speech_duration += raw_duration

        if raw_duration < min_speech_seconds:
            analyzed_segments.append(
                {
                    **segment,
                    "duration": round(raw_duration, 3),
                    "risk_level": "INSUFFICIENT_AUDIO",
                    "reason": f"segment shorter than {min_speech_seconds:.2f}s",
                }
            )
            continue

        processed = preprocess_samples(raw, sample_rate, audio_config)
        if not processed.valid or processed.duration_seconds < min_speech_seconds:
            analyzed_segments.append(
                {
                    **segment,
                    "duration": round(raw_duration, 3),
                    "usable_duration": round(processed.duration_seconds, 3),
                    "risk_level": "INSUFFICIENT_AUDIO",
                    "reason": processed.warning or "segment too short after preprocessing",
                }
            )
            continue

        result = model.predict_samples(
            processed.samples,
            processed.sample_rate,
            audio_info={
                "speaker_id": speaker_id,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "duration_seconds": processed.duration_seconds,
            },
        )
        risk_level = risk_level_from_fake_score(float(result["fake_score"]), result["thresholds"])
        segment_result = {
            **segment,
            "duration": round(raw_duration, 3),
            "usable_duration": round(processed.duration_seconds, 3),
            "real_score": float(result["real_score"]),
            "fake_score": float(result["fake_score"]),
            "risk_score": float(result["risk_score"]),
            "risk_level": risk_level,
            "model_verdict": result["verdict"],
        }
        analyzed_segments.append(segment_result)
        reliable_results.append(segment_result)

    if not reliable_results:
        return _empty_speaker_result(
            speaker_id,
            analyzed_segments,
            f"no speaker segment had at least {min_speech_seconds:.2f}s of usable speech",
        )

    weights = np.asarray([float(item["usable_duration"]) for item in reliable_results], dtype=np.float32)
    fake_scores = np.asarray([float(item["fake_score"]) for item in reliable_results], dtype=np.float32)
    real_scores = np.asarray([float(item["real_score"]) for item in reliable_results], dtype=np.float32)
    weighted_fake = float(np.average(fake_scores, weights=weights))
    weighted_real = float(np.average(real_scores, weights=weights))
    max_fake = float(fake_scores.max())
    thresholds = model.metadata["decision_thresholds"]
    risk_level = risk_level_from_fake_score(weighted_fake, thresholds)

    high_risk_segments = sum(item["risk_level"] == "HIGH RISK" for item in reliable_results)
    if max_fake >= STRONG_FAKE_SCORE and high_risk_segments >= MIN_HIGH_RISK_SEGMENTS_FOR_CALL:
        risk_level = "HIGH RISK"

    return {
        "speaker_id": speaker_id,
        "speech_duration": round(total_speech_duration, 3),
        "usable_speech_duration": round(float(weights.sum()), 3),
        "segment_count": len(segments),
        "reliable_segment_count": len(reliable_results),
        "real_score": weighted_real,
        "fake_score": weighted_fake,
        "risk_score": weighted_fake,
        "max_fake_score": max_fake,
        "risk_level": risk_level,
        "status": risk_level,
        "high_risk_segments": high_risk_segments,
        "segments": analyzed_segments,
    }


def _group_by_speaker(segments: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for segment in segments:
        grouped[str(segment["speaker"])].append(segment)
    return {speaker: sorted(items, key=lambda item: float(item["start"])) for speaker, items in grouped.items()}


def _aggregate_call_risk(
    speaker_results: List[Dict[str, object]],
    thresholds: Dict[str, object],
    min_reliable_speaker_seconds: float = MIN_RELIABLE_SPEAKER_SECONDS,
) -> Dict[str, object]:
    reliable = [
        speaker
        for speaker in speaker_results
        if speaker["risk_level"] != "INSUFFICIENT_AUDIO"
        and float(speaker.get("usable_speech_duration", 0.0)) >= min_reliable_speaker_seconds
    ]

    if not reliable:
        return {
            "overall_risk_score": 0.0,
            "overall_risk_level": "INSUFFICIENT_AUDIO",
            "reason": "No speaker had enough usable speech for reliable VoiceShield analysis.",
        }

    strongest = max(reliable, key=lambda speaker: float(speaker["fake_score"]))
    high_risk_speakers = [speaker for speaker in reliable if speaker["risk_level"] == "HIGH RISK"]
    suspicious_speakers = [speaker for speaker in reliable if speaker["risk_level"] == "SUSPICIOUS"]
    fake_threshold = float(thresholds["fake_if_fake_score_at_or_above"])

    if high_risk_speakers:
        return {
            "overall_risk_score": float(strongest["fake_score"]),
            "overall_risk_level": "HIGH RISK",
            "reason": f"High synthetic-speech probability detected for {strongest['speaker_id']}.",
        }

    if suspicious_speakers or float(strongest["fake_score"]) >= fake_threshold:
        return {
            "overall_risk_score": float(strongest["fake_score"]),
            "overall_risk_level": "SUSPICIOUS",
            "reason": "One or more reliable speakers landed in the suspicious score range.",
        }

    return {
        "overall_risk_score": float(strongest["fake_score"]),
        "overall_risk_level": "SAFE",
        "reason": "No reliable speaker exceeded the suspicious or high-risk thresholds.",
    }


def analyze_multispeaker_call(
    audio_path: str | Path,
    *,
    model: Optional[VoiceShieldModel] = None,
    min_speech_seconds: float = MIN_SPEECH_SECONDS,
    fallback_to_chunk_detector: bool = True,
) -> Dict[str, object]:
    path = Path(audio_path)
    messages: List[str] = []

    try:
        diarization_segments = diarize_audio(path)
    except (DiarizationSetupError, FileNotFoundError, RuntimeError, ValueError) as exc:
        if not fallback_to_chunk_detector:
            raise
        fallback = analyze_chunked_audio_file(path)
        fallback["fallback_used"] = True
        fallback["fallback_reason"] = f"speaker diarization failed: {exc}"
        fallback["analysis_type"] = "chunk_fallback"
        fallback["detected_speakers"] = None
        fallback["overall_risk_level"] = fallback["risk_level"]
        fallback["speakers"] = []
        return fallback

    if not diarization_segments:
        if not fallback_to_chunk_detector:
            return {
                "audio_path": str(path),
                "detected_speakers": 0,
                "overall_risk_score": 0.0,
                "overall_risk_level": "INSUFFICIENT_AUDIO",
                "speakers": [],
                "messages": ["No speech segments detected by diarization."],
                "fallback_used": False,
            }
        fallback = analyze_chunked_audio_file(path)
        fallback["fallback_used"] = True
        fallback["fallback_reason"] = "no speech segments detected by diarization"
        fallback["analysis_type"] = "chunk_fallback"
        fallback["detected_speakers"] = 0
        fallback["overall_risk_level"] = fallback["risk_level"]
        fallback["speakers"] = []
        return fallback

    grouped = _group_by_speaker(diarization_segments)
    if len(grouped) == 1:
        messages.append("Only one diarized speaker detected; speaker-level analysis still used.")

    model = model or VoiceShieldModel()
    raw_samples, sample_rate = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    raw_samples = raw_samples.astype(np.float32, copy=False)

    speaker_results = [
        _analyze_speaker_segments(
            speaker_id,
            segments,
            raw_samples,
            sample_rate,
            model,
            min_speech_seconds=min_speech_seconds,
        )
        for speaker_id, segments in sorted(grouped.items())
    ]

    thresholds = model.metadata["decision_thresholds"]
    call_summary = _aggregate_call_risk(speaker_results, thresholds)
    total_audio = load_audio(path)

    return {
        "audio_path": str(path),
        "analysis_type": "speaker_diarization",
        "fallback_used": False,
        "detected_speakers": len(grouped),
        "total_duration": total_audio.duration_seconds,
        "sample_rate": sample_rate,
        "min_speech_seconds": min_speech_seconds,
        "thresholds": thresholds,
        "overall_risk_score": call_summary["overall_risk_score"],
        "overall_risk_level": call_summary["overall_risk_level"],
        "reason": call_summary["reason"],
        "messages": messages,
        "diarization_segments": diarization_segments,
        "speakers": speaker_results,
    }


def percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def print_analysis(analysis: Dict[str, object]) -> None:
    if analysis.get("analysis_type") == "chunk_fallback":
        print("\nVOICE SHIELD MULTI-SPEAKER ANALYSIS")
        print("-----------------------------------")
        print("Fallback used:", analysis.get("fallback_reason"))
        print("Overall Call Risk Score:", percent(float(analysis["overall_risk_score"])))
        print("Overall Call Status:", analysis["risk_level"])
        return

    print("\n## VOICE SHIELD MULTI-SPEAKER ANALYSIS")
    print("File:", analysis["audio_path"])
    print("Detected speakers:", analysis["detected_speakers"])
    for message in analysis.get("messages", []):
        print("Note:", message)

    for speaker in analysis["speakers"]:
        print(f"\n{speaker['speaker_id']}")
        print("Duration:", f"{speaker['speech_duration']:.2f} sec")
        print("Segments:", speaker["segment_count"])
        print("Reliable segments:", speaker["reliable_segment_count"])
        print("REAL:", percent(speaker["real_score"]))
        print("FAKE:", percent(speaker["fake_score"]))
        print("Status:", speaker["risk_level"])
        if speaker["risk_level"] == "INSUFFICIENT_AUDIO":
            print("Reason:", speaker["reason"])

    print("\nOverall Call Risk Score:", percent(float(analysis["overall_risk_score"])))
    print("Overall Call Status:", analysis["overall_risk_level"])
    print("Reason:", analysis["reason"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VoiceShield multi-speaker call analysis.")
    parser.add_argument("audio_file")
    parser.add_argument("--min-speech-seconds", type=float, default=MIN_SPEECH_SECONDS)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    analysis = analyze_multispeaker_call(
        args.audio_file,
        min_speech_seconds=args.min_speech_seconds,
        fallback_to_chunk_detector=not args.no_fallback,
    )
    print_analysis(analysis)


if __name__ == "__main__":
    main()
