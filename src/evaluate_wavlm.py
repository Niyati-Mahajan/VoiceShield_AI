import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from voiceshield_inference import VoiceShieldModel
from wavlm_features import supported_audio_files


LABEL_NAMES = ["REAL", "FAKE"]


def collect_testing_files() -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for label_name, label_value in [("real", 0), ("fake", 1)]:
        folder = Path("dataset/testing") / label_name
        for audio_path in supported_audio_files(folder):
            items.append((audio_path, label_value))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VoiceShield on untouched dataset/testing.")
    parser.add_argument("--no-cache", action="store_true", help="Disable testing embedding cache.")
    args = parser.parse_args()

    model = VoiceShieldModel()
    items = collect_testing_files()
    print(f"Testing files: {len(items)}")

    y_true = []
    y_pred = []
    verdicts = {"REAL": 0, "SUSPICIOUS": 0, "FAKE": 0}

    for index, (audio_path, label) in enumerate(items, start=1):
        cache_dir = None if args.no_cache else Path("cache/wavlm/testing")
        result = model.predict_file(audio_path, cache_dir=cache_dir)
        y_true.append(label)
        y_pred.append(result["binary_label"])
        verdicts[result["verdict"]] += 1
        print(
            f"{index}/{len(items)} {audio_path} true={LABEL_NAMES[label]} "
            f"binary={result['binary_prediction']} verdict={result['verdict']} "
            f"fake_score={result['fake_score']:.4f}"
        )

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1])
    real_fp_rate = matrix[0, 1] / max(1, matrix[0].sum())
    fake_fn_rate = matrix[1, 0] / max(1, matrix[1].sum())

    print("\nDATASET TEST RESULTS")
    print("--------------------")
    print(f"Accuracy: {accuracy_score(y_true_array, y_pred_array):.4f}")
    print(classification_report(y_true_array, y_pred_array, target_names=LABEL_NAMES, zero_division=0))
    print("Confusion matrix [[REAL->REAL, REAL->FAKE], [FAKE->REAL, FAKE->FAKE]]:")
    print(matrix)
    print(f"REAL false-positive rate: {real_fp_rate:.4f}")
    print(f"FAKE false-negative rate: {fake_fn_rate:.4f}")
    print("Three-way verdict counts:", verdicts)


if __name__ == "__main__":
    main()
