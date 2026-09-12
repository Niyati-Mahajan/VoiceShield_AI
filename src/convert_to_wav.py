import librosa
import soundfile as sf

input_file = "test_audio/Recording.m4a"
output_file = "test_audio/human_test.wav"

audio, sr = librosa.load(
    input_file,
    sr=16000,
    mono=True
)

sf.write(
    output_file,
    audio,
    16000
)

print("Converted successfully!")
print("Saved as:", output_file)