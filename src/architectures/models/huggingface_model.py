from typing import Dict, Union
import os
import math

import torch
from torch import nn

from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        is_preprocessed: bool,
        custom_data_encoder_path: str,
        left_padding: bool,
        merged_model_path: str,
        precision: Union[int, str],
        mode: str,
        quantization_type: str,
        quantization_config: BitsAndBytesConfig,
        peft_type: str,
        peft_config: LoraConfig,
        num_labels: int,
        system: int,
    ) -> None:
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.is_preprocessed = is_preprocessed
        if is_preprocessed:
            data_encoder_path = custom_data_encoder_path
        else:
            data_encoder_path = self.pretrained_model_name
        self.data_encoder = AutoTokenizer.from_pretrained(
            data_encoder_path,
            use_fast=True,
        )
        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id
        if left_padding:
            self.data_encoder.padding_side = "left"

        self.merged_model_path = merged_model_path
        self.model_path = self.pretrained_model_name
        if self.is_preprocessed:
            if os.path.exists(f"{self.merged_model_path}/{self.pretrained_model_name}"):
                self.model_path = (
                    f"{self.merged_model_path}/{self.pretrained_model_name}"
                )

        self.attn_implementation = None
        if precision == 32 or precision == "32":
            self.precision = torch.float32
        elif precision == 16 or precision == "16":
            self.precision = torch.float16
            self.attn_implementation = "flash_attention_2"
        elif precision == "bf16":
            self.precision = torch.bfloat16
            self.attn_implementation = "flash_attention_2"
        else:
            self.precision = "auto"

        self.mode = mode
        self.quantization_type = quantization_type
        self.quantization_config = None
        self.device_map = None
        if self.quantization_type == "quantization":
            self.quantization_config = quantization_config
            if self.mode in ["test" "predict"]:
                self.quantization_config.load_in_4bit = False
            else:
                self.quantization_config.load_in_4bit = True
            self.quantization_config.bnb_4bit_compute_dtype = self.precision
            self.device_map = {
                "": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))
            }
        if self.quantization_type not in ["origin", "quantization"]:
            raise ValueError(f"Invalid quantization type: {self.quantization_type}.")

        self.peft_type = peft_type
        self.peft_config = peft_config
        if self.mode in ["test" "predict"]:
            self.peft_config.inference_mode = True

        self.num_labels = num_labels
        self.system = system
        self.num_digits = round(
            math.log(
                self.num_labels,
                self.system,
            )
        )

        self.model = self.get_model()
        self.score = nn.Linear(
            self.model.config.hidden_size,
            self.system,
            bias=False,
        )

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = self.model(**encoded)
        logits = self.score(output.hidden_states[0])
        input_ids = encoded["input_ids"]
        batch_size = input_ids.shape[0]
        sequence_lengths = (
            torch.eq(
                input_ids,
                self.model.config.pad_token_id,
            )
            .int()
            .argmax(-1)
            - 1
        )
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
        logits_of_digits = [
            logits[
                torch.arange(
                    batch_size,
                    device=logits.device,
                ),
                sequence_lengths - i,
            ]
            for i in range(self.num_digits)
        ]
        return {
            "logits_of_digits": logits_of_digits,
        }

    def get_model(self) -> PreTrainedModel:
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            output_hidden_states=True,
            torch_dtype=self.precision,
            attn_implementation=self.attn_implementation,
            quantization_config=self.quantization_config,
            device_map=self.device_map,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )

        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(self.data_encoder))

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

        if self.quantization_type == "quantization" and self.mode not in [
            "test",
            "predict",
        ]:
            model = prepare_model_for_kbit_training(model)

        if self.peft_type == "lora":
            model.enable_input_require_grads()
            model = get_peft_model(model, self.peft_config)
        if self.peft_type not in ["origin", "lora"]:
            raise ValueError(f"Invalid PEFT type: {self.peft_type}.")
        return model
