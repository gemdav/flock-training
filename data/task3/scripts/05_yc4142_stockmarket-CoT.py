from datasets import load_dataset
import json
import os


OUTPUT_PATH = f"data/task3/{os.path.basename(__file__)}.jsonl"

# load dataset
ds = load_dataset("yc4142/stockmarket-CoT")

# convert dataset and write to file
with open(OUTPUT_PATH, "w") as f:
    for split, subset in ds.items():
        for item in subset:
            formatted_item = {
                "conversations": [
                    {
                        "role": "user",
                        "content": item["input"].strip(),
                    }, {
                        "role": "assistant",
                        "content": item["output"].strip()
                    }
                ],
                "system": item["instruction"].strip()
            }
            f.write(json.dumps(formatted_item) + "\n")  # Write in JSONL format
