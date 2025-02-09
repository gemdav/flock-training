import json
import os
from datasets import load_dataset


OUTPUT_PATH = f"data/task3/{os.path.basename(__file__)}.jsonl"

# load dataset
ds = load_dataset("fine-tuned/jinaai_jina-embeddings-v2-base-en-6122024-bhm2-webapp")

# convert dataset and write to file
with open(OUTPUT_PATH, "w") as f:
    for split, subset in ds.items():
        for item in subset:

            if item["query"].endswith("?"):
                query = item["query"]
                response = item["pos"][0]
            elif item["pos"][0].endswith("?"):
                query = item["pos"][0]
                response = item["query"]
            else:
                continue

            formatted_item = {
                "conversations": [
                    {
                        "role": "user",
                        "content": query
                    }, {
                        "role": "assistant",
                        "content": response
                    }
                ],
                "system": "You are a helpful financial assistant."
            }
            f.write(json.dumps(formatted_item) + "\n")  # Write in JSONL format
