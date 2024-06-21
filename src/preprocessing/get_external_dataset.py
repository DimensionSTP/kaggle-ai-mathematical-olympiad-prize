import dotenv

dotenv.load_dotenv(
    override=True,
)

from huggingface_hub import hf_hub_download

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def get_external_dataset(
    config: DictConfig,
) -> None:
    hf_hub_download(
        repo_id="camel-ai/math",
        repo_type="dataset",
        filename="math.zip",
        local_dir=f"{config.connected_dir}/data",
        local_dir_use_symlinks=False,
    )


if __name__ == "__main__":
    get_external_dataset()
