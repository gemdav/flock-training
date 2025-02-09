import json
import re
import requests
import os
from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()
OUTPUT_PATH = f"data/task3/{os.path.basename(__file__)}.jsonl"
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
DEEPSEEK_API_URL = "https://ai-defaultaihub518257332458.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
DEPLOYMENT_NAME = "DeepSeek-R1"

client = AzureOpenAI(
    azure_endpoint=DEEPSEEK_API_URL,
    api_key=AZURE_API_KEY,
    api_version="2024-05-01-preview",
)

# Define the dataset files and their GitHub URLs
dataset_files = {
    "dev": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "private_test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/private_test.json",
    "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
}

# Download each file and store its content in the merged_data dictionary
for split, url in dataset_files.items():
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if download fails
    split = response.json()

    for item in split:
        chat_prompt = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            },
            {
                "role": "user",
                "content": f'''
I will provide you with a pre-text of a table, the table itself, and a post-text of the table. 
Afterwards I will give you a question and an answer that can be given based on the table, the pre-text and the post-text.
Your task is to formulate a question and an answer in full sentences based on my input such that the information from the table needed to answer the question is contained in the question. It is important that you provide all the necessary information, also the one given in the table, the pre-text, and the post-text, in the question itself! Write questions and answers of at least 30 words.

Pre-text: {item["pre_text"]}

Post-text: {item["pre_text"]}

Table: {item["table"]}

Question: {item["qa"]["question"]}

Answer: {item["qa"]["answer"]}

provide you answer in JSON format with "question" and "answer" as fields.
'''
            }
        ]

        # Include speech result if speech is enabled
        completion = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=chat_prompt,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        completion = json.loads(completion.to_json())

        question_match = re.search(r'"question": "(.*?)",\n  "answer": ', completion["choices"][0]["message"]["content"], re.DOTALL)
        answer_match = re.search(r'",\n  "answer": "(.*?)"\n}', completion["choices"][0]["message"]["content"], re.DOTALL)
        question = question_match.group(1) if question_match else None
        answer = answer_match.group(1) if answer_match else None

        if question is None or answer is None:
            continue

        data = {
            "conversations": [
                {
                    "role": "user",
                    "content": question,
                }, {
                    "role": "assistant",
                    "content": answer
                }
            ],
            "system": "You are an expert in the financial field and are good at answering financial-related questions raised by everyone."
        }

        # Ensure the file exists before appending
        if not os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
                pass  # Create an empty file

        # Append the dictionary as a new JSON line
        with open(OUTPUT_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(data) + "\n")
