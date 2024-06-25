from typing import Dict, Any, List
import math

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class KaggleMathOlympiadDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        is_preprocessed: bool,
        data_column_name: str,
        prompt_column_name: str,
        target_column_name: str,
        num_devices: int,
        batch_size: int,
        pretrained_model_name: str,
        custom_data_encoder_path: str,
        data_max_length: int,
        target_max_length: int,
        num_labels: int,
        system: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.is_preprocessed = is_preprocessed
        self.data_column_name = data_column_name
        self.prompt_column_name = prompt_column_name
        self.target_column_name = target_column_name
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.pretrained_model_name = pretrained_model_name
        if self.is_preprocessed:
            data_encoder_path = (
                f"{custom_data_encoder_path}/{self.pretrained_model_name}"
            )
        else:
            data_encoder_path = self.pretrained_model_name
        self.data_encoder = AutoTokenizer.from_pretrained(
            data_encoder_path,
            use_fast=True,
        )
        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id
        dataset = self.get_dataset()
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]
        self.data_max_length = data_max_length
        self.target_max_length = target_max_length
        self.num_labels = num_labels
        self.system = system
        self.num_digits = round(
            math.log(
                self.num_labels,
                self.system,
            )
        )
        self.original_train_df = pd.read_csv(f"{self.data_path}/train.csv")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        if self.is_preprocessed:
            prompt = self.datas[idx] + self.labels[idx]
        else:
            prompt = self.generate_prompt(
                data=self.datas[idx],
                label=self.labels[idx],
            )
        encoded = self.encode_text(
            data=prompt,
            data_type="data",
        )
        encoded["labels"] = torch.tensor(
            [self.labels[idx]],
            dtype=torch.long,
        ).squeeze(0)
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        str_labels = f"{self.labels[idx]:0{self.num_digits}d}"
        labels_per_digit = [int(str_label) for str_label in str_labels]
        labels_per_digit.reverse()
        labels_per_digit = torch.tensor(
            labels_per_digit,
            dtype=torch.long,
        )
        return {
            "encoded": encoded,
            "labels_per_digit": labels_per_digit,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            if self.is_preprocessed:
                parquet_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/train.parquet"
            else:
                parquet_path = f"{self.data_path}/math-ai/TemplateGSM/train.parquet"
            data = pd.read_parquet(parquet_path)
            data = data.fillna("_")
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/{self.split}.csv"
            else:
                csv_path = f"{self.data_path}/{self.split}.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
        elif self.split == "predict":
            if self.is_preprocessed:
                csv_path = f"{self.data_path}/preprocessed_dataset/{self.pretrained_model_name}/test.csv"
            else:
                csv_path = f"{self.data_path}/test.csv"
            data = pd.read_csv(csv_path)
            data = data.fillna("_")
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")
        if self.is_preprocessed:
            datas = data[self.prompt_column_name].tolist()
        else:
            datas = data[self.data_column_name].apply(lambda x: x.strip()).tolist()
        labels = data[self.target_column_name].tolist()
        return {
            "datas": datas,
            "labels": labels,
        }

    def encode_text(
        self,
        data: str,
        data_type: str,
    ) -> Dict[str, torch.Tensor]:
        if data_type == "data":
            if self.split == "predict":
                max_length = self.data_max_length
            else:
                max_length = self.data_max_length + self.target_max_length
        elif data_type == "target":
            max_length = self.target_max_length
        else:
            raise ValueError(f"Inavalid data_type: {data_type}")
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def generate_prompt(
        self,
        data: str,
        label: str,
    ) -> str:
        default_system_prompt = f"""
You are an advanced math problem-solving expert. Your task is to accurately understand the given math problem and provide the answer. The answer to each problem is an integer between 0 and 999. Do not provide explanations, only the final integer answer. Follow these rules:

1. **Problem Understanding**: Understand the problem accurately.
2. **Provide the Answer**: Directly provide the final integer answer.
3. **Answer Verification**: Ensure that your answer is correct and falls within the range of 0 to 999.
4. **No Explanation**: Do not provide step-by-step explanations or any additional information. Only provide the final answer.

### Example Problems:

**example 1**:
problem:
{self.original_train_df[self.data_column_name][3]}
answer:
{self.original_train_df[self.target_column_name][3]}

**example 2**:
problem:
{self.original_train_df[self.data_column_name][4]}
answer:
{self.original_train_df[self.target_column_name][4]}

**example 3**:
problem:
{self.original_train_df[self.data_column_name][5]}
answer:
{self.original_train_df[self.target_column_name][5]}
"""
        if self.split == "predict":
            prompt = f"""### Instruction:
{default_system_prompt} 

### Input(problem):
{data.strip()}

### Response(answer):
""".strip()
        else:
            prompt = f"""### Instruction:
{default_system_prompt} 

### Input(problem):
{data.strip()}

### Response(answer):
{label} """.strip()
        return prompt
