import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import math

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="voting.yaml",
)
def softly_vote_logits(
    config: DictConfig,
) -> None:
    connected_dir = config.connected_dir

    num_digits = round(
        math.log(
            config.num_labels,
            config.system,
        )
    )

    voted_logit = config.voted_logit
    submission_file = config.submission_file
    data_column_name = config.data_column_name
    target_column_name = config.target_column_name
    voted_file = config.voted_file
    votings = config.votings

    weights = list(votings.values())
    if not np.isclose(sum(weights), 1):
        raise ValueError(f"summation of weights({sum(weights)}) is not equal to 1")

    ensemble_predictions_per_digit = []
    for i in range(num_digits):
        weighted_logits = None
        for logit_file, weight in votings.items():
            try:
                logit = np.load(f"{connected_dir}/logits/{logit_file}.npy")
            except:
                raise FileNotFoundError(f"logit file {logit_file} does not exist")
            if weighted_logits is None:
                weighted_logits = logit * weight
            else:
                weighted_logits += logit * weight

        ensemble_predictions = np.argmax(
            weighted_logits,
            axis=-1,
        )
        np.save(
            f"{voted_logit}-digit_{i}.npy",
            weighted_logits,
        )
        ensemble_predictions_per_digit.append(ensemble_predictions)

    ensemble_predictions_weighted_sum = np.zeros_like(ensemble_predictions_per_digit[0])
    for i, ensemble_predictions in enumerate(ensemble_predictions_per_digit):
        digit_weight = 10**i
        ensemble_predictions_weighted_sum += digit_weight * ensemble_predictions

    submission_df = pd.read_csv(submission_file)
    submission_df[target_column_name] = ensemble_predictions_weighted_sum
    submission_df = submission_df.drop(
        data_column_name,
        axis=1,
    )
    submission_df.to_csv(
        voted_file,
        index=False,
    )


if __name__ == "__main__":
    softly_vote_logits()
