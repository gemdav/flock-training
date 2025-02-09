import os
from dataclasses import dataclass
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from core.sft import SFTDataset, SFTDataCollator
from utils.constants import MODEL2TEMPLATE


@dataclass
class Lora:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int

    def train(self, model_id: str, context_length: int, data_path: str):
        assert model_id in MODEL2TEMPLATE, f"model_id {model_id} not supported"
        lora_config = LoraConfig(
            r=self.lora_rank,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            task_type="CAUSAL_LM",
        )

        # load model in 4-bit to do qLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        sft_args = SFTConfig(
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=100,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=20,
            output_dir="../outputs",
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
            num_train_epochs=self.num_train_epochs,
            max_seq_length=context_length,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            token=os.environ["HF_TOKEN"],
        )

        # load dataset
        dataset = SFTDataset(
            filepath=data_path,
            tokenizer=tokenizer,
            max_seq_length=context_length,
            template=MODEL2TEMPLATE[model_id],
        )

        # define trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=sft_args,
            peft_config=lora_config,
            data_collator=SFTDataCollator(
                tokenizer, max_seq_length=context_length),
        )

        # train model
        trainer.train()

        # save model
        trainer.save_model("outputs")

        # remove checkpoint folder
        os.system("rm -rf ../outputs/checkpoint-*")

        # upload lora weights and tokenizer
        print("Training Completed.")
