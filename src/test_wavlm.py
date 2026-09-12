import argparse

from voiceshield_inference import VoiceShieldModel


def print_result(result: dict) -> None:
    print("\nVOICE ANALYSIS RESULT")
    print("---------------------")
    print("File:", result["audio_path"])
    print("Verdict:", result["verdict"])
    print("Binary model prediction:", result["binary_prediction"])
    print("Real score:", round(result["real_score"] * 100, 2), "%")
    print("Fake score:", round(result["fake_score"] * 100, 2), "%")
    print("Risk score:", round(result["risk_score"] * 100, 2), "%")
    print("Confidence:", round(result["confidence"] * 100, 2), "%")
    print("Decision thresholds:", result["thresholds"])
    print("Audio duration after preprocessing:", round(result["audio_info"]["duration_seconds"], 2), "s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test one audio file with VoiceShield WavLM.")
    parser.add_argument("audio_file")
    args = parser.parse_args()

    model = VoiceShieldModel()
    result = model.predict_file(args.audio_file)
    print_result(result)


if __name__ == "__main__":
    main()
