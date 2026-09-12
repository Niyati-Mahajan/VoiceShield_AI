import librosa
import soundfile as sf

input_file = "test_audio/Recording_4.wav"
output_file = "test_audio/Recording_4_trimmed.wav"

audio, sr = librosa.load(
    input_file,
    sr=16000,
    mono=True
)

trimmed_audio, _ = librosa.effects.trim(
    audio,
    top_db=30
)

sf.write(
    output_file,
    trimmed_audio,
    sr
)

print("Original duration:", round(len(audio) / sr, 2), "seconds")
print("Trimmed duration:", round(len(trimmed_audio) / sr, 2), "seconds")
print("Saved:", output_file)