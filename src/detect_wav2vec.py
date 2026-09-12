import sys
import librosa
import torch
import numpy as np

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
TARGET_SR = 16000


print("Loading model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)

model.eval()

print("Model loaded successfully!")


def load_audio(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    return audio


def detect_voice(audio_path):

    audio = load_audio(audio_path)

    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)[0]

    print("\nMODEL LABELS:")
    print(model.config.id2label)

    print("\nRAW PROBABILITIES:")
    for i, prob in enumerate(probabilities):
        label = model.config.id2label.get(i, str(i))
        print(label, ":", round(prob.item() * 100, 2), "%")

    predicted_class = torch.argmax(probabilities).item()

    prediction = model.config.id2label[predicted_class]

    confidence = probabilities[predicted_class].item()

    print("\nVOICE ANALYSIS RESULT")
    print("---------------------")
    print("Prediction:", prediction)
    print("Confidence:", round(confidence * 100, 2), "%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python src/detect_wav2vec.py <audio_file>")
        sys.exit()

    audio_file = sys.argv[1]

    detect_voice(audio_file)