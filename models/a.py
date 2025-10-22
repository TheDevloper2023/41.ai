import os
import json

# Root folder containing all models
models_root = r"C:/Users/L/Desktop/41.ai-rewrite/models"

# Output JSON file
output_json = os.path.join(models_root, "models.json")

# Load existing JSON if present
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {"models": {}}

# Loop through every subfolder in the models directory
for folder in os.listdir(models_root):
    model_path = os.path.join(models_root, folder)
    if not os.path.isdir(model_path):
        continue

    # Look for Tacotron2 checkpoint
    taco_path = os.path.join(model_path, "taco.pt")
    if os.path.exists(taco_path):
        model_name = folder
        speaker_ids = os.path.join(model_path, "speaker_id.txt")
        vocoder_path = os.path.join(model_path, "vocoder")

        data["models"][model_name] = {
            "Checkpoint": taco_path.replace("\\", "/"),
            "Architecture": "Tacotron2",
            "Vocoder": vocoder_path.replace("\\", "/") if os.path.exists(vocoder_path) else "",
            "SpeakerIDS": speaker_ids.replace("\\", "/") if os.path.exists(speaker_ids) else ""
        }

# Save updated JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Model list updated! Saved to {output_json}")
