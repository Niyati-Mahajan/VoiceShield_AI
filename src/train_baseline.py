import os
import librosa
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump


TARGET_SR = 16000


def extract_features(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)

    # RMS energy
    rms = librosa.feature.rms(y=audio)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [np.mean(spectral_centroid)],
        [np.std(spectral_centroid)],
        [np.mean(spectral_bandwidth)],
        [np.std(spectral_bandwidth)],
        [np.mean(spectral_rolloff)],
        [np.std(spectral_rolloff)],
        [np.mean(zcr)],
        [np.std(zcr)],
        [np.mean(rms)],
        [np.std(rms)],
    ])

    return features


def load_dataset(split):
    X = []
    y = []

    base_path = os.path.join("dataset", split)

    for label_name, label_value in [
        ("real", 0),
        ("fake", 1)
    ]:

        folder = os.path.join(
            base_path,
            label_name
        )

        for filename in os.listdir(folder):

            if not filename.lower().endswith(".wav"):
                continue

            file_path = os.path.join(
                folder,
                filename
            )

            try:
                features = extract_features(file_path)

                X.append(features)
                y.append(label_value)

            except Exception as e:
                print(
                    f"Error processing {file_path}: {e}"
                )

    return np.array(X), np.array(y)


print("\nLoading training data...")

X_train, y_train = load_dataset("training")

print("Training samples:", len(X_train))


print("\nLoading validation data...")

X_val, y_val = load_dataset("validation")

print("Validation samples:", len(X_val))


model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])


print("\nTraining VoiceShield baseline model...")

model.fit(
    X_train,
    y_train
)


print("\nEvaluating on validation data...")

predictions = model.predict(X_val)

accuracy = accuracy_score(
    y_val,
    predictions
)

print(
    "\nValidation Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        predictions,
        target_names=[
            "REAL",
            "FAKE"
        ]
    )
)


os.makedirs(
    "models/voiceshield_baseline",
    exist_ok=True
)

dump(
    model,
    "models/voiceshield_baseline/model.joblib"
)

print(
    "\nModel saved successfully!"
)