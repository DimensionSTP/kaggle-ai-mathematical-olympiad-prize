import dotenv

dotenv.load_dotenv(
    override=True,
)

import glob

import pandas as pd

from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def preprocess_external_dataset(
    config: DictConfig,
) -> None:
    path = f"{config.connected_dir}/data/math-ai/TemplateGSM"
    file_paths = glob.glob(f"{path}/*.parquet")
    processed_dfs = []
    min_value = 0
    max_value = config.num_labels - 1
    for file_path in tqdm(file_paths):
        df = pd.read_parquet(file_path)
        df = df.rename(
            columns={
                "result": config.target_column_name,
            }
        )
        df[config.target_column_name] = pd.to_numeric(
            df[config.target_column_name],
            errors="coerce",
        )
        df = df[
            df[config.target_column_name]
            == df[config.target_column_name].astype(
                int,
                errors="ignore",
            )
        ]
        df[config.target_column_name] = df[config.target_column_name].astype(int)
        df = df[
            (df[config.target_column_name] >= min_value)
            & (df[config.target_column_name] <= max_value)
        ]
        sampled_df = (
            df.groupby("template_id")
            .apply(
                lambda x: x.sample(
                    n=min(
                        5,
                        len(x),
                    ),
                    random_state=config.seed,
                )
            )
            .reset_index(drop=True)
        )
        processed_dfs.append(sampled_df)
    final_df = pd.concat(
        processed_dfs,
        ignore_index=True,
    )
    final_df.to_parquet(
        f"{path}/sample_and_concat.parquet",
        index=False,
    )


if __name__ == "__main__":
    preprocess_external_dataset()
