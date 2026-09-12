import sys
import librosa
import numpy as np
from joblib import load

TARGET_SR = 16000
MODEL_PATH = "models/voiceshield_baseline/model.joblib"

print("Loading model...")

model = load(MODEL_PATH)

print("Model loaded successfully!")


def extract_features(audio_path):
    print("Loading audio:", audio_path)

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    print("Audio loaded successfully!")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(audio)

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

    return features.reshape(1, -1)


def predict_audio(audio_path):
    features = extract_features(audio_path)

    print("Running prediction...")

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    real_probability = probabilities[0]
    fake_probability = probabilities[1]

    label = "REAL" if prediction == 0 else "FAKE"

    print("\nVOICE ANALYSIS RESULT")
    print("---------------------")
    print("Prediction:", label)
    print("Real score:", round(real_probability * 100, 2), "%")
    print("Fake score:", round(fake_probability * 100, 2), "%")


if __name__ == "__main__":

    print("Script started")

    if len(sys.argv) < 2:
        print("Usage:")
        print("python src/test_baseline.py <audio_file>")
        sys.exit()

    audio_file = sys.argv[1]

    print("Selected file:", audio_file)

    predict_audio(audio_file)