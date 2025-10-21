import json
import os

# Path to your models.json
json_path = r"C:\Users\L\Desktop\41.ai-rewrite\tts\main_config.json"

# The new vocoder path to set for all models
new_vocoder_path = r"C:/Users/L/Desktop/41.ai-rewrite/models/SheZow/vocoder"

# Load existing JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Replace vocoder path for every model
for model_name, model_info in data.get("models", {}).items():
    model_info["Vocoder"] = new_vocoder_path.replace("\\", "/")

# Save updated JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ All vocoder paths updated to: {new_vocoder_path}")
