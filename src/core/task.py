from datetime import datetime
from dataclasses import dataclass
import json
import requests

from utils.constants import FED_LEDGER_BASE_URL


@dataclass
class Task:
    title: str
    description: str
    task_type: str
    training_set_url: str
    max_params: int
    context_length: int
    duration_in_seconds: int
    id: int
    status: str
    initialized_at: datetime
    submission_phase_ends_at: datetime
    final_validation_ends_at: datetime
    final_link: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            title=data["title"],
            description=data["description"],
            task_type=data["task_type"],
            training_set_url=data["data"]["training_set_url"],
            max_params=data["data"]["max_params"],
            context_length=data["data"]["context_length"],
            duration_in_seconds=data["duration_in_seconds"],
            id=data["id"],
            status=data["status"],
            initialized_at=datetime.fromisoformat(data["initialized_at"]),
            submission_phase_ends_at=datetime.fromisoformat(
                data["submission_phase_ends_at"]),
            final_validation_ends_at=datetime.fromisoformat(
                data["final_validation_ends_at"]),
            final_link=data.get("final_link")  # Optional, so it can be None
        )

    @classmethod
    def get(cls, task_id: int):
        response = requests.request(
            "GET", f"{FED_LEDGER_BASE_URL}/tasks/get?task_id={task_id}", timeout=30)
        return cls.from_dict(response.json())

    def submit(self, hf_repo_id: str, base_model: str, gpu_type: str, revision: str, api_key: str):
        payload = json.dumps(
            {
                "task_id": self.id,
                "data": {
                    "hg_repo_id": hf_repo_id,
                    "base_model": base_model,
                    "gpu_type": gpu_type,
                    "revision": revision,
                },
            }
        )
        headers = {
            "flock-api-key": api_key,
            "Content-Type": "application/json",
        }
        response = requests.request(
            "POST",
            f"{FED_LEDGER_BASE_URL}/tasks/submit-result",
            headers=headers,
            data=payload,
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to submit task: {response.text}")
        print("Task submitted successfully!")
        return response.json()

    def get_demo_data(self, target_path: str):
        # download in chunks
        response = requests.get(
            self.training_set_url, stream=True, timeout=30)
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Downloaded data successfully!")
