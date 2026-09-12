import os
import random
import shutil

random.seed(42)

SOURCE = r"C:\Users\SAMSUNG\Downloads\for-2sec\for-2seconds"
DESTINATION = "dataset"

SAMPLES = {
    "training": 500,
    "validation": 100,
    "testing": 100
}

for split, number_of_samples in SAMPLES.items():

    for label in ["real", "fake"]:

        source_folder = os.path.join(SOURCE, split, label)
        destination_folder = os.path.join(
            DESTINATION,
            split,
            label
        )

        os.makedirs(destination_folder, exist_ok=True)

        files = [
            file
            for file in os.listdir(source_folder)
            if file.lower().endswith(".wav")
        ]

        selected_files = random.sample(
            files,
            min(number_of_samples, len(files))
        )

        for i, filename in enumerate(selected_files, start=1):

            source_file = os.path.join(
                source_folder,
                filename
            )

            new_filename = f"{label}_{i:03d}.wav"

            destination_file = os.path.join(
                destination_folder,
                new_filename
            )

            shutil.copy2(
                source_file,
                destination_file
            )

        print(
            f"{split}: copied {len(selected_files)} "
            f"{label} samples"
        )

print("\nVoiceShield mini dataset created successfully!")