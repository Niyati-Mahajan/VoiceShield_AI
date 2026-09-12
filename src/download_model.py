from huggingface_hub import hf_hub_download
import os

repo_id = "SpeechAntiSpoofingBenchmarks/AASIST"
model_dir = "models/AASIST"

os.makedirs(model_dir, exist_ok=True)

files = [
    "AASIST.pth",
    "aasist.onnx",
]

for file in files:
    print(f"Downloading {file}...")

    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        local_dir=model_dir
    )

print("\nAASIST model files downloaded successfully!")