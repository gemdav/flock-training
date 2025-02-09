import json
import os
from datasets import load_dataset


OUTPUT_PATH = f"data/task3/{os.path.basename(__file__)}.jsonl"

# load dataset
ds = load_dataset("innodatalabs/rt-inod-finance")

# convert dataset and write to file
with open(OUTPUT_PATH, "w") as f:
    for split, subset in ds.items():
        for item in subset:

            if "Translate" in item["messages"][1]["content"] or "Transcribe" in item["messages"][1]["content"]:
                continue
            if "Sorry, I cannot help" in item["expected"]:
                continue

            formatted_item = {
                "conversations": [
                    {
                        "role": "user",
                        "content": item["messages"][1]["content"]
                        .replace("Here's a question or series of instructions:\n\n", "", 1).strip()
                        .replace("\n\nIf the question or instructions above are innapropriate or outside of financial domain, start your answer with 'Sorry, I cannot help':", "", 1).strip(),
                    }, {
                        "role": "assistant",
                        "content": item["expected"]
                    }
                ],
                "system": "You are a helpful financial assistant."
            }
            f.write(json.dumps(formatted_item) + "\n")  # Write in JSONL format
