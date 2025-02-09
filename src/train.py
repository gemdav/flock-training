import json
import os
import yaml
from loguru import logger
from huggingface_hub import HfApi
from dotenv import load_dotenv

from core.task import Task
from core.lora import Lora
from utils.constants import MODEL2BASE_MODEL, MODEL2SIZE
from utils.gpu_utils import get_gpu_type

# load environment variables
load_dotenv()
HF_USERNAME = os.environ["HF_USERNAME"]
HF_TOKEN = os.environ["HF_TOKEN"]
TASK_ID = os.environ["TASK_ID"]
FLOCK_API_KEY = os.environ["FLOCK_API_KEY"]


def main():
    # get training arguments
    with open("./training_args.yaml", "r", encoding="utf-8") as f:
        all_training_args = yaml.safe_load(f)

    # get task
    task = Task.get(TASK_ID)
    logger.info(json.dumps(task.__dict__, indent=4, default=str))

    # filter out the model within the max_params
    model2size_filtered = {
        k: v for k, v in MODEL2SIZE.items() if v <= task.max_params}
    all_training_args = {
        k: v for k, v in all_training_args.items() if k in model2size_filtered}
    logger.info(f"Models within the max_params: {all_training_args.keys()}")

    # train all feasible models and merge
    for model_id in all_training_args.keys():
        logger.info(f"Start to train the model {model_id}...")

        # train model
        try:
            Lora(**all_training_args[model_id]).train(
                model_id=model_id,
                context_length=task.context_length,
                data_path="../data/_data.jsonl"
            )
        except RuntimeError as e:
            # if OOM, proceed to the next model
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
            continue

        # create a HF repo and submit the task
        try:
            logger.info("Start to push the lora weight to the hub...")

            hf_api = HfApi(token=HF_TOKEN)
            hf_repo_name = f"{HF_USERNAME}/flockio-task-{TASK_ID}-{model_id.replace('/', '-')}"

            # create repo if it does not exist
            try:
                hf_api.create_repo(
                    hf_repo_name,
                    exist_ok=False,
                    repo_type="model",
                )
            except Exception:
                logger.info(
                    f"Repo {hf_repo_name} already exists. Will commit the new version."
                )

            # commit the model
            commit_message = hf_api.upload_folder(
                folder_path="./outputs",
                repo_id=hf_repo_name,
                repo_type="model",
            )

            # get commit hash
            commit_hash = commit_message.oid
            logger.info(f"Commit hash: {commit_hash}")
            logger.info(f"Repo name: {hf_repo_name}")

            # submit task to flock api
            task.submit(
                hf_repo_id=hf_repo_name,
                base_model=MODEL2BASE_MODEL[model_id],
                gpu_type=get_gpu_type(),
                revision=commit_hash,
                api_key=FLOCK_API_KEY
            )
            logger.info("Task submitted successfully")

        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")

        finally:
            # cleanup merged_model and output
            os.system("rm -rf merged_model")
            os.system("rm -rf ./outputs")
            continue


if __name__ == "__main__":
    main()
