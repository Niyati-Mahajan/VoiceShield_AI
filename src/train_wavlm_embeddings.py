import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from wavlm_features import MODEL_ID, WavLMFeatureExtractor, cached_embedding, supported_audio_files


LABELS = {"REAL": 0, "FAKE": 1}
LABEL_NAMES = ["REAL", "FAKE"]
MODEL_DIR = Path("models/voiceshield_wavlm")
CACHE_ROOT = Path("cache/wavlm")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_split(split: str, include_custom: bool = False) -> List[Tuple[Path, int, str]]:
    items: List[Tuple[Path, int, str]] = []
    sources = [(Path("dataset") / split, split)]
    if include_custom:
        sources.append((Path("custom_data"), "custom"))

    for base_path, source_name in sources:
        for label_name, label_value in [("real", 0), ("fake", 1)]:
            folder = base_path / label_name
            for audio_path in supported_audio_files(folder):
                items.append((audio_path, label_value, source_name))
    return items


def warn_if_dataset_leakage() -> None:
    split_hashes: Dict[str, Dict[str, Path]] = {}
    for split in ["training", "validation", "testing"]:
        split_hashes[split] = {}
        for audio_path, _, _ in collect_split(split):
            split_hashes[split][file_sha256(audio_path)] = audio_path

    for left, right in [("training", "validation"), ("training", "testing"), ("validation", "testing")]:
        overlap = set(split_hashes[left]).intersection(split_hashes[right])
        if overlap:
            print(f"WARNING: possible leakage between {left} and {right}: {len(overlap)} duplicate files")


def load_embeddings(
    extractor: WavLMFeatureExtractor,
    items: List[Tuple[Path, int, str]],
    split: str,
    *,
    augment_copies: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []

    for index, (audio_path, label, source_name) in enumerate(items, start=1):
        cache_dir = CACHE_ROOT / ("custom" if source_name == "custom" else split)
        try:
            embedding, _, hit = cached_embedding(extractor, audio_path, cache_dir)
            X.append(embedding)
            y.append(label)
            cache_status = "cache" if hit else "new"
            print(f"{index}/{len(items)} {audio_path} [{cache_status}]")

            if split == "training" and augment_copies > 0:
                for copy_index in range(augment_copies):
                    aug_seed = seed + index * 1000 + copy_index
                    variant = f"aug{copy_index + 1}-seed{aug_seed}"
                    aug_embedding, _, aug_hit = cached_embedding(
                        extractor,
                        audio_path,
                        cache_dir,
                        variant=variant,
                        augment=True,
                        seed=aug_seed,
                    )
                    X.append(aug_embedding)
                    y.append(label)
                    aug_status = "cache" if aug_hit else "new"
                    print(f"  augmentation {copy_index + 1}/{augment_copies} [{aug_status}]")
        except Exception as exc:
            print(f"WARNING: skipped {audio_path}: {exc}")

    if not X:
        raise RuntimeError(f"No usable audio files found for {split}")

    return np.vstack(X), np.asarray(y, dtype=np.int64)


def classifier_candidates() -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
            ]
        ),
        "linear_svm_calibrated": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(class_weight="balanced", random_state=42, max_iter=5000),
                        cv=3,
                    ),
                ),
            ]
        ),
        "small_mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        alpha=1e-3,
                        early_stopping=True,
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def fake_probabilities(model: Pipeline, X: np.ndarray) -> np.ndarray:
    classes = list(model.named_steps["classifier"].classes_)
    fake_index = classes.index(1)
    return model.predict_proba(X)[:, fake_index]


def validation_thresholds(model: Pipeline, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float | bool]:
    fake_scores = fake_probabilities(model, X_val)
    real_scores_on_real = fake_scores[y_val == 0]
    fake_scores_on_fake = fake_scores[y_val == 1]

    real_threshold = float(np.percentile(fake_scores_on_fake, 5)) if len(fake_scores_on_fake) else 0.40
    fake_threshold = float(np.percentile(real_scores_on_real, 95)) if len(real_scores_on_real) else 0.60
    real_threshold = min(real_threshold, 0.45)
    fake_threshold = max(fake_threshold, 0.55)

    if real_threshold >= fake_threshold:
        real_threshold = 0.40
        fake_threshold = 0.60

    return {
        "real_if_fake_score_at_or_below": round(real_threshold, 6),
        "fake_if_fake_score_at_or_above": round(fake_threshold, 6),
        "suspicious_between_thresholds": True,
    }


def print_metrics(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    real_fp_rate = matrix[0, 1] / max(1, matrix[0].sum())
    fake_fn_rate = matrix[1, 0] / max(1, matrix[1].sum())

    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))
    print("Confusion matrix [[REAL->REAL, REAL->FAKE], [FAKE->REAL, FAKE->FAKE]]:")
    print(matrix)
    print(f"REAL false-positive rate: {real_fp_rate:.4f}")
    print(f"FAKE false-negative rate: {fake_fn_rate:.4f}")

    return {
        "accuracy": float(accuracy),
        "precision_real": float(precision[0]),
        "recall_real": float(recall[0]),
        "f1_real": float(f1[0]),
        "precision_fake": float(precision[1]),
        "recall_fake": float(recall[1]),
        "f1_fake": float(f1[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": matrix.tolist(),
        "real_false_positive_rate": float(real_fp_rate),
        "fake_false_negative_rate": float(fake_fn_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VoiceShield WavLM embedding classifiers.")
    parser.add_argument("--augment-copies", type=int, default=1, help="Training-only augmented copies per file.")
    parser.add_argument("--no-custom", action="store_true", help="Ignore custom_data/real and custom_data/fake.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print("Label mapping: REAL=0, FAKE=1")
    warn_if_dataset_leakage()

    training_items = collect_split("training", include_custom=not args.no_custom)
    validation_items = collect_split("validation", include_custom=False)
    print(f"Training source files: {len(training_items)}")
    print(f"Validation source files: {len(validation_items)}")

    extractor = WavLMFeatureExtractor(model_id=MODEL_ID)
    print(f"Using device: {extractor.device}")
    print("Extracting/loading training embeddings...")
    X_train, y_train = load_embeddings(
        extractor,
        training_items,
        "training",
        augment_copies=max(0, args.augment_copies),
        seed=args.seed,
    )
    print("Extracting/loading validation embeddings...")
    X_val, y_val = load_embeddings(extractor, validation_items, "validation")

    print(f"Training matrix: {X_train.shape}")
    print(f"Validation matrix: {X_val.shape}")

    results: Dict[str, Dict[str, object]] = {}
    best_name = ""
    best_model = None
    best_score = -1.0

    for name, model in classifier_candidates().items():
        print(f"\nTraining classifier: {name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        metrics = print_metrics(f"Validation results: {name}", y_val, predictions)
        results[name] = metrics
        selection_score = float(metrics["macro_f1"]) + 0.10 * float(metrics["recall_real"])
        if selection_score > best_score:
            best_score = selection_score
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No classifier was trained")

    thresholds = validation_thresholds(best_model, X_val, y_val)
    print(f"\nSelected classifier: {best_name}")
    print(f"Decision thresholds: {thresholds}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "classifier.joblib")

    metadata = {
        "project": "VoiceShield",
        "label_mapping": LABELS,
        "selected_classifier": best_name,
        "candidate_validation_metrics": results,
        "decision_thresholds": thresholds,
        "feature_config": extractor.config_dict(),
        "training": {
            "training_source_files": len(training_items),
            "validation_source_files": len(validation_items),
            "augment_copies": max(0, args.augment_copies),
            "custom_data_included": not args.no_custom,
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved classifier and metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
