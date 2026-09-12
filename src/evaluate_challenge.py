from pathlib import Path

from voiceshield_inference import VoiceShieldModel


CHALLENGE_SET = [
    (Path("test_audio/human_test.wav"), "REAL"),
    (Path("test_audio/ai_test.wav"), "FAKE"),
    (Path("test_audio/ai_test_2.wav"), "FAKE"),
]


def main() -> None:
    model = VoiceShieldModel()

    print("CHALLENGE SET EVALUATION")
    print("------------------------")
    print("These files are evaluation-only and are never loaded by train_wavlm_embeddings.py.")

    correct_binary = 0
    for audio_path, true_label in CHALLENGE_SET:
        result = model.predict_file(audio_path)
        binary_prediction = result["binary_prediction"]
        correct_binary += int(binary_prediction == true_label)
        print(
            f"{audio_path}: true={true_label} binary={binary_prediction} "
            f"verdict={result['verdict']} real_score={result['real_score']:.4f} "
            f"fake_score={result['fake_score']:.4f} risk_score={result['risk_score']:.4f}"
        )

    print(f"\nBinary challenge accuracy: {correct_binary}/{len(CHALLENGE_SET)}")


if __name__ == "__main__":
    main()
