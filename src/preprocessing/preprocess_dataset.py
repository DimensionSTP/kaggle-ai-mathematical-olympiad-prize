import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

import pandas as pd

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def preprocess_dataset(
    config: DictConfig,
) -> None:
    if config.mode == "train":
        df = pd.read_parquet(
            f"{config.connected_dir}/data/{config.external_data_path}/{config.mode}.parquet"
        )
    elif config.mode == "test":
        df = pd.read_csv(f"{config.connected_dir}/data/{config.mode}.csv")
    else:
        raise ValueError(f"Invalid mode: {config.mode}")
    original_train_df = pd.read_csv(f"{config.connected_dir}/data/train.csv")
    tokenizer = AutoTokenizer.from_pretrained(
        f"{config.custom_data_encoder_path}/{config.pretrained_model_name}",
        use_fast=True,
    )

    def generate_prompt(
        data: str,
    ) -> str:
        default_system_prompt = f"""
You are an advanced math problem-solving expert. Your task is to accurately understand the given math problem and provide the answer. The answer to each problem is an integer between 000 and 999. Do not provide explanations, only the final integer answer. Follow these rules:

1. **Problem Understanding**: Understand the problem accurately.
2. **Provide the Answer**: Directly provide the final integer answer.
3. **Answer Verification**: Ensure that your answer is correct, always expressed as a three-digit number, and falls within the range of 000 to 999.
4. **No Explanation**: Do not provide step-by-step explanations or any additional information. Only provide the final answer.

### Example Problems:

**example 1**:
problem:
{original_train_df[config.data_column_name][3]}
answer:
{original_train_df[config.target_column_name][3]:03d}

**example 2**:
problem:
{original_train_df[config.data_column_name][4]}
answer:
{original_train_df[config.target_column_name][4]:03d}

**example 3**:
problem:
{original_train_df[config.data_column_name][5]}
answer:
{original_train_df[config.target_column_name][5]:03d}
"""
        prompt = f"""### Instruction:
{default_system_prompt} 

### Input(problem):
{data.strip()}

### Response(answer):
""".strip()
        return prompt

    df["prompt"] = df[config.data_column_name].apply(generate_prompt)
    df[config.data_column_name] = df[config.data_column_name].apply(lambda x: x.strip())

    def cut_prompt_to_length(
        prompt: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> str:
        tokens = tokenizer.tokenize(prompt)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        cut_prompt = tokenizer.convert_tokens_to_string(tokens)
        return cut_prompt

    df[config.prompt_column_name] = df["prompt"].apply(
        lambda x: cut_prompt_to_length(
            prompt=x,
            tokenizer=tokenizer,
            max_length=config.data_max_length,
        )
    )
    if not os.path.exists(
        f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}"
    ):
        os.makedirs(
            f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}",
            exist_ok=True,
        )
    if config.mode == "train":
        df.to_parquet(
            f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}/{config.mode}.parquet",
            index=False,
        )
    else:
        df.to_csv(
            f"{config.connected_dir}/data/preprocessed_dataset/{config.pretrained_model_name}/{config.mode}.csv",
            index=False,
        )


if __name__ == "__main__":
    preprocess_dataset()
