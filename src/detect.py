import librosa
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/AASIST/aasist.onnx"
TARGET_SR = 16000
TARGET_LENGTH = 64600

session = ort.InferenceSession(MODEL_PATH)


def prepare_audio(audio_path):
    # Load audio as mono and resample to 16 kHz
    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    # Convert to float32
    audio = audio.astype(np.float32)

    # If audio is shorter than 64600 samples, repeat it
    if len(audio) < TARGET_LENGTH:
        repeats = int(np.ceil(TARGET_LENGTH / len(audio)))
        audio = np.tile(audio, repeats)

    # Keep first 64600 samples
    audio = audio[:TARGET_LENGTH]

    # Add batch dimension: (64600,) -> (1, 64600)
    audio = np.expand_dims(audio, axis=0)

    return audio


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


def detect_voice(audio_path):

    audio = prepare_audio(audio_path)

    outputs = session.run(
        ["logits"],
        {"wav": audio}
    )

    logits = outputs[0][0]

    probabilities = softmax(logits)

    spoof_probability = float(probabilities[0])
    human_probability = float(probabilities[1])

    if human_probability > spoof_probability:
        prediction = "HUMAN"
        confidence = human_probability
    else:
        prediction = "AI_GENERATED"
        confidence = spoof_probability

    return {
        "prediction": prediction,
        "human_probability": human_probability,
        "spoof_probability": spoof_probability,
        "confidence": confidence,
        "raw_logits": logits.tolist()
    }


if __name__ == "__main__":

    audio_file = "test_audio/ai_test_2.wav"

    result = detect_voice(audio_file)

    print("\nVOICE ANALYSIS RESULT")
    print("---------------------")

    print("Prediction:", result["prediction"])

    print(
        "Human Probability:",
        round(result["human_probability"] * 100, 2),
        "%"
    )

    print(
        "AI Probability:",
        round(result["spoof_probability"] * 100, 2),
        "%"
    )

    print(
        "Confidence:",
        round(result["confidence"] * 100, 2),
        "%"
    )

    print("Raw logits:", result["raw_logits"])