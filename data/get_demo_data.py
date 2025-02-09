import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from core.task import Task  # noqa

# load environment variables
load_dotenv()
TASK_ID = os.environ["TASK_ID"]


def main():
    directory = f"data/task{TASK_ID}"
    os.makedirs(directory, exist_ok=True)
    Task.get(TASK_ID).get_demo_data(f"{directory}/demo_data.jsonl")


if __name__ == "__main__":
    main()
