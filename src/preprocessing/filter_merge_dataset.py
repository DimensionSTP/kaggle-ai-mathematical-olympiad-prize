import dotenv

dotenv.load_dotenv(
    override=True,
)

import pandas as pd

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def filter_merge_dataset(
    config: DictConfig,
) -> None:
    df = pd.read_parquet(
        f"{config.connected_dir}/data/math-ai/TemplateGSM/sample_and_concat.parquet"
    )
    original_train_df = pd.read_csv(f"{config.connected_dir}/data/train.csv")
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        use_fast=True,
    )

    def generate_prompt(
        data: str,
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
{original_train_df[config.data_column_name][3]}
answer:
{original_train_df[config.target_column_name][3]}

**example 2**:
problem:
{original_train_df[config.data_column_name][4]}
answer:
{original_train_df[config.target_column_name][4]}

**example 3**:
problem:
{original_train_df[config.data_column_name][5]}
answer:
{original_train_df[config.target_column_name][5]}
"""
        prompt = f"""### Instruction:
{default_system_prompt} 

### Input(problem):
{data.strip()}

### Response(answer):
""".strip()
        return prompt

    df["prompt"] = df[config.data_column_name].apply(generate_prompt)
    df[f"tokenized_{config.data_column_name}"] = df[config.data_column_name].apply(
        lambda x: tokenizer.tokenize(x)
    )
    df[f"token_{config.data_column_name}_length"] = df[
        f"tokenized_{config.data_column_name}"
    ].apply(len)
    df["tokenized_prompt"] = df["prompt"].apply(lambda x: tokenizer.tokenize(x))
    df["token_prompt_length"] = df["tokenized_prompt"].apply(len)
    df[f"tokenized_{config.target_column_name}"] = df[config.target_column_name].apply(
        lambda x: tokenizer.tokenize(str(x))
    )
    df[f"token_{config.target_column_name}_length"] = df[
        f"tokenized_{config.target_column_name}"
    ].apply(len)
    df["token_total_length"] = (
        df["token_prompt_length"] + df[f"token_{config.target_column_name}_length"]
    )
    df_filtered = df[
        df["token_prompt_length"] <= tokenizer.config.model_max_length
    ].drop(columns=["token_prompt_length"])
    df_filtered = df_filtered[
        [
            config.data_column_name,
            config.target_column_name,
        ]
    ]
    original_train_df_dropped = original_train_df_dropped[
        [
            config.data_column_name,
            config.target_column_name,
        ]
    ]
    original_train_df_dropped[config.target_column_name] = original_train_df_dropped[
        config.target_column_name
    ].astype(int)
    df_concat = pd.concat(
        [
            df_filtered,
            original_train_df_dropped,
        ],
        axis=0,
        ignore_index=True,
    )
    df_concat.to_parquet(
        f"{config.connected_dir}/data/math-ai/TemplateGSM/train.parquet",
        index=False,
    )


if __name__ == "__main__":
    filter_merge_dataset()
