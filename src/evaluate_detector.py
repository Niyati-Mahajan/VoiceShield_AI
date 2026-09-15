import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from voiceshield_inference import VoiceShieldModel


DEFAULT_CHALLENGE_DIR = Path("challenge_set")
DEFAULT_RESULTS_DIR = Path("results")
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
LABEL_TO_ID = {"REAL": 0, "FAKE": 1}


def find_challenge_files(challenge_dir: Path) -> List[Dict[str, object]]:
    """Find unseen challenge audio and attach ground-truth labels from folders."""
    items: List[Dict[str, object]] = []

    for audio_path in sorted(challenge_dir.rglob("*")):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue

        relative_parts = audio_path.relative_to(challenge_dir).parts
        if not relative_parts:
            continue

        top_folder = relative_parts[0].lower()
        if top_folder == "real":
            ground_truth = "REAL"
        elif top_folder == "fake":
            ground_truth = "FAKE"
        else:
            continue

        subcategory = "/".join(relative_parts[:2]) if len(relative_parts) >= 2 else top_folder
        items.append(
            {
                "file": audio_path,
                "category": top_folder,
                "subcategory": subcategory.replace("\\", "/"),
                "ground_truth": ground_truth,
            }
        )

    return items


def evaluate_file(model: VoiceShieldModel, item: Dict[str, object], cache_dir: Optional[Path]) -> Dict[str, object]:
    """Run the existing VoiceShield model on one challenge file."""
    audio_path = Path(item["file"])
    ground_truth = str(item["ground_truth"])

    try:
        result = model.predict_file(audio_path, cache_dir=cache_dir)
        prediction = str(result["binary_prediction"])
        correct = prediction == ground_truth

        return {
            "file": str(audio_path),
            "filename": audio_path.name,
            "category": item["category"],
            "subcategory": item["subcategory"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "verdict": result.get("verdict", prediction),
            "real_score": float(result["real_score"]),
            "fake_score": float(result["fake_score"]),
            "confidence": float(result["confidence"]),
            "correct": correct,
            "error": "",
        }
    except Exception as exc:
        return {
            "file": str(audio_path),
            "filename": audio_path.name,
            "category": item["category"],
            "subcategory": item["subcategory"],
            "ground_truth": ground_truth,
            "prediction": "ERROR",
            "verdict": "ERROR",
            "real_score": "",
            "fake_score": "",
            "confidence": "",
            "correct": False,
            "error": str(exc),
        }


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def calculate_eer(y_true: np.ndarray, fake_scores: np.ndarray) -> Optional[float]:
    """Calculate equal error rate from FAKE probability scores when possible."""
    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, fake_scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def binary_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    valid_rows = [row for row in rows if row["prediction"] in LABEL_TO_ID]
    failed_rows = [row for row in rows if row["prediction"] == "ERROR"]

    if not valid_rows:
        return {
            "evaluated_samples": 0,
            "failed_samples": len(failed_rows),
            "accuracy": 0.0,
            "precision_fake": 0.0,
            "recall_fake": 0.0,
            "f1_fake": 0.0,
            "confusion_matrix": {
                "real_predicted_real": 0,
                "real_predicted_fake": 0,
                "fake_predicted_real": 0,
                "fake_predicted_fake": 0,
            },
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "roc_auc": None,
            "eer": None,
        }

    y_true = np.asarray([LABEL_TO_ID[str(row["ground_truth"])] for row in valid_rows])
    y_pred = np.asarray([LABEL_TO_ID[str(row["prediction"])] for row in valid_rows])
    fake_scores = np.asarray([float(row["fake_score"]) for row in valid_rows], dtype=np.float32)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = int(matrix[0, 0])
    fp = int(matrix[0, 1])
    fn = int(matrix[1, 0])
    tp = int(matrix[1, 1])

    roc_auc = None
    eer = None
    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, fake_scores))
        eer = calculate_eer(y_true, fake_scores)

    return {
        "evaluated_samples": len(valid_rows),
        "failed_samples": len(failed_rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": {
            "real_predicted_real": tn,
            "real_predicted_fake": fp,
            "fake_predicted_real": fn,
            "fake_predicted_fake": tp,
        },
        "false_positive_rate": safe_rate(fp, tn + fp),
        "false_negative_rate": safe_rate(fn, tp + fn),
        "roc_auc": roc_auc,
        "eer": eer,
    }


def subcategory_summary(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["subcategory"])].append(row)

    summary: Dict[str, Dict[str, object]] = {}
    for subcategory, group_rows in sorted(groups.items()):
        metrics = binary_metrics(group_rows)
        summary[subcategory] = {
            "total_samples": len(group_rows),
            **metrics,
        }
    return summary


def save_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "filename",
        "category",
        "subcategory",
        "ground_truth",
        "prediction",
        "verdict",
        "real_score",
        "fake_score",
        "confidence",
        "correct",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_percent(name: str, value: Optional[float]) -> None:
    if value is None:
        print(f"{name}: N/A")
    else:
        print(f"{name}: {value * 100:.2f}%")


def print_error_rows(rows: List[Dict[str, object]]) -> None:
    failed_rows = [row for row in rows if row["prediction"] == "ERROR"]
    if not failed_rows:
        return

    print("\nFAILED / CORRUPT FILES")
    for row in failed_rows:
        print(f"{row['file']} | true={row['ground_truth']} | error={row['error']}")


def print_false_positives(rows: List[Dict[str, object]]) -> None:
    print("\nFALSE POSITIVES")
    print("REAL audio predicted as FAKE")
    false_positives = [
        row
        for row in rows
        if row["ground_truth"] == "REAL" and row["prediction"] == "FAKE"
    ]
    if not false_positives:
        print("None")
        return

    for row in false_positives:
        print(f"{row['file']} | subcategory={row['subcategory']} | fake_score={float(row['fake_score']):.4f}")


def print_false_negatives(rows: List[Dict[str, object]]) -> None:
    print("\nFALSE NEGATIVES")
    print("FAKE audio predicted as REAL")
    false_negatives = [
        row
        for row in rows
        if row["ground_truth"] == "FAKE" and row["prediction"] == "REAL"
    ]
    if not false_negatives:
        print("None")
        return

    for row in false_negatives:
        print(f"{row['file']} | subcategory={row['subcategory']} | fake_score={float(row['fake_score']):.4f}")


def false_positive_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if row["ground_truth"] == "REAL" and row["prediction"] == "FAKE"
    ]


def false_negative_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if row["ground_truth"] == "FAKE" and row["prediction"] == "REAL"
    ]


def print_summary(summary: Dict[str, object]) -> None:
    metrics = summary["metrics"]
    matrix = metrics["confusion_matrix"]

    print("\nVOICE SHIELD DETECTOR EVALUATION")
    print("--------------------------------")
    print("Challenge directory:", summary["challenge_dir"])
    print("Total samples found:", summary["total_samples"])
    print("Samples evaluated:", metrics["evaluated_samples"])
    print("Failed samples:", metrics["failed_samples"])
    print_percent("Accuracy", metrics["accuracy"])
    print_percent("Precision for FAKE", metrics["precision_fake"])
    print_percent("Recall for FAKE", metrics["recall_fake"])
    print_percent("F1 score for FAKE", metrics["f1_fake"])
    print("Confusion matrix [[REAL->REAL, REAL->FAKE], [FAKE->REAL, FAKE->FAKE]]:")
    print(
        [
            [matrix["real_predicted_real"], matrix["real_predicted_fake"]],
            [matrix["fake_predicted_real"], matrix["fake_predicted_fake"]],
        ]
    )
    print_percent("False-positive rate", metrics["false_positive_rate"])
    print_percent("False-negative rate", metrics["false_negative_rate"])
    if metrics["roc_auc"] is None:
        print("ROC-AUC: N/A")
    else:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print_percent("EER", metrics["eer"])

    print("\nSUBCATEGORY PERFORMANCE")
    for subcategory, sub_metrics in summary["by_subcategory"].items():
        print(
            f"{subcategory}: total={sub_metrics['total_samples']} "
            f"evaluated={sub_metrics['evaluated_samples']} "
            f"accuracy={sub_metrics['accuracy'] * 100:.2f}% "
            f"failed={sub_metrics['failed_samples']}"
        )


def evaluate_challenge_set(challenge_dir: Path, results_dir: Path, use_cache: bool = True) -> Dict[str, object]:
    items = find_challenge_files(challenge_dir)
    model = VoiceShieldModel()
    cache_dir = Path("cache/wavlm/challenge") if use_cache else None

    rows: List[Dict[str, object]] = []
    for index, item in enumerate(items, start=1):
        row = evaluate_file(model, item, cache_dir)
        rows.append(row)
        print(
            f"{index}/{len(items)} {row['file']} true={row['ground_truth']} "
            f"pred={row['prediction']} verdict={row['verdict']} correct={row['correct']}"
        )

    summary = {
        "challenge_dir": str(challenge_dir),
        "results_csv": str(results_dir / "challenge_results.csv"),
        "summary_json": str(results_dir / "challenge_summary.json"),
        "total_samples": len(items),
        "label_mapping": model.metadata.get("label_mapping", {}),
        "decision_thresholds": model.metadata.get("decision_thresholds", {}),
        "metrics": binary_metrics(rows),
        "by_subcategory": subcategory_summary(rows),
        "false_positives": false_positive_rows(rows),
        "false_negatives": false_negative_rows(rows),
        "failed_files": [row for row in rows if row["prediction"] == "ERROR"],
    }

    save_csv(rows, results_dir / "challenge_results.csv")
    save_json(summary, results_dir / "challenge_summary.json")

    print_summary(summary)
    print_false_positives(rows)
    print_false_negatives(rows)
    print_error_rows(rows)
    print("\nSaved detailed results:", results_dir / "challenge_results.csv")
    print("Saved summary:", results_dir / "challenge_summary.json")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VoiceShield on unseen challenge_set data.")
    parser.add_argument("--challenge-dir", default=str(DEFAULT_CHALLENGE_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--no-cache", action="store_true", help="Disable challenge embedding cache.")
    args = parser.parse_args()

    challenge_dir = Path(args.challenge_dir)
    results_dir = Path(args.results_dir)

    if not challenge_dir.exists():
        raise SystemExit(
            f"Challenge directory not found: {challenge_dir}\n\n"
            "Create the unseen evaluation folder at the project root with this layout:\n"
            "challenge_set/\n"
            "  real/\n"
            "    english/\n"
            "    hindi/\n"
            "    punjabi/\n"
            "  fake/\n"
            "    generator_1/\n"
            "    generator_2/\n"
            "    generator_3/\n\n"
            "Then copy unseen test audio files into those folders and run this command again:\n"
            "python src\\evaluate_detector.py"
        )

    evaluate_challenge_set(
        challenge_dir=challenge_dir,
        results_dir=results_dir,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
