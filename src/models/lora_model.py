"""
use LoRA finetuning model
"""

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pooling import (
    MeanPooling,
    MeanPoolingProjection,
    Attention1dPoolingHead,
    LightAttentionPoolingHead,
    MeanPoolingHead,
)
from src.utils.lora_tools import prepare_for_lora_model
from transformers import EsmModel, BertModel, T5EncoderModel


def get_plm_model(plm_model):
    if "esm" in plm_model:
        plm_model = EsmModel.from_pretrained(plm_model)
    elif "bert" in plm_model:
        plm_model = BertModel.from_pretrained(plm_model)
    elif "prot_t5" in plm_model:
        plm_model = T5EncoderModel.from_pretrained(plm_model)
    elif "ankh" in plm_model:
        plm_model = T5EncoderModel.from_pretrained(plm_model)
    return plm_model


class Model(nn.Module):
    """
    finetuning encoder
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.plm_model = get_plm_model(config.plm_model)
        # print("self.plm_model ", self.plm_model)
        if "esm" in config.plm_model:
            config.hidden_size = self.plm_model.config.hidden_size
        elif "bert" in config.plm_model:
            config.hidden_size = self.plm_model.config.hidden_size
        elif "prot_t5" in config.plm_model:
            config.hidden_size = self.plm_model.config.d_model
        elif "ankh" in config.plm_model:
            config.hidden_size = self.plm_model.config.d_model

        if config.pooling_method == "attention1d":
            self.classifier = Attention1dPoolingHead(
                config.hidden_size, config.num_labels, config.pooling_dropout
            )
        elif config.pooling_method == "mean":
            if "PPI" in config.dataset:
                self.pooling = MeanPooling()
                self.projection = MeanPoolingProjection(
                    config.hidden_size, config.num_labels, config.pooling_dropout
                )
            else:
                self.classifier = MeanPoolingHead(
                    config.hidden_size, config.num_labels, config.pooling_dropout
                )
        elif config.pooling_method == "light_attention":
            self.classifier = LightAttentionPoolingHead(
                config.hidden_size, config.num_labels, config.pooling_dropout
            )
        else:
            raise ValueError(f"classifier method {config.pooling_method} not supported")

        if self.config.use_lora:
            target_modules = getattr(
                config, "lora_target_modules", ["key", "query", "value"]
            )
            self.plm_model = prepare_for_lora_model(
                self.plm_model,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
            )
        else:
            # freeze encoder
            for param in self.plm_model.parameters():
                param.requires_grad = False

    def forward(self, batch):
        aa_seq, attention_mask = batch["aa_input_ids"], batch["attention_mask"]
        outputs = self.plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        embeds = outputs.last_hidden_state

        logits = self.classifier(embeds, attention_mask)
        return logits

    # def save_model(self, save_path):
    #     """save LoRA and model params"""
    #     os.makedirs(save_path, exist_ok=True)

    #     self.plm_model.save_pretrained(os.path.join(save_path, "lora_weights"))

    #     classifier_path = os.path.join(save_path, "classifier.pt")
    #     if hasattr(self, "classifier"):
    #         torch.save(self.classifier.state_dict(), classifier_path)
    #     else:
    #         pooling_path = os.path.join(save_path, "pooling.pt")
    #         projection_path = os.path.join(save_path, "projection.pt")
    #         torch.save(self.pooling.state_dict(), pooling_path)
    #         torch.save(self.projection.state_dict(), projection_path)

    # @classmethod
    # def load_model(cls, config, load_path):
    #     """load model params"""
    #     model = cls(config)

    #     model.plm_model = PeftModel.from_pretrained(
    #         model.plm_model, os.path.join(load_path, "lora_weights")
    #     )
        
    #     if hasattr(model, "classifier"):
    #         classifier_path = os.path.join(load_path, "classifier.pt")
    #         model.classifier.load_state_dict(torch.load(classifier_path))
    #     else:
    #         pooling_path = os.path.join(load_path, "pooling.pt")
    #         projection_path = os.path.join(load_path, "projection.pt")
    #         model.pooling.load_state_dict(torch.load(pooling_path))
    #         model.projection.load_state_dict(torch.load(projection_path))

    #     return model
