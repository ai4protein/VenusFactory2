"""
use LoRA finetuning model
"""
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .pooling import Attention1dPoolingHead, MeanPoolingHead, LightAttentionPoolingHead
from .pooling import MeanPooling, MeanPoolingProjection


class LoraModel(nn.Module):
    """
    finetuning encoder
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        if args.pooling_method == "attention1d":
            self.classifier = Attention1dPoolingHead(
                args.hidden_size, args.num_labels, args.pooling_dropout
            )
        elif args.pooling_method == "mean":
            if "PPI" in args.dataset:
                self.pooling = MeanPooling()
                self.projection = MeanPoolingProjection(
                    args.hidden_size, args.num_labels, args.pooling_dropout
                )
            else:
                self.classifier = MeanPoolingHead(
                    args.hidden_size, args.num_labels, args.pooling_dropout
                )
        elif args.pooling_method == "light_attention":
            self.classifier = LightAttentionPoolingHead(
                args.hidden_size, args.num_labels, args.pooling_dropout
            )
        else:
            raise ValueError(f"classifier method {args.pooling_method} not supported")

    def plm_embedding(self, plm_model, aa_seq, attention_mask, stru_token):
        if (
            self.training
            and hasattr(self, "args")
            and self.args.training_method in ["full", "lora", "plm-lora"]
        ):
            if "ProSST" in self.args.plm_model:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, ss_input_ids=stru_token, output_hidden_states=True)
            else:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        if "ProSST" in self.args.plm_model:
            seq_embeds = outputs.hidden_states[-1]
        else:
            seq_embeds = outputs.last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds

    def forward(self, plm_model, batch):
        aa_seq, attention_mask, stru_token = (
            batch["aa_seq_input_ids"],
            batch["aa_seq_attention_mask"],
            batch["aa_seq_stru_tokens"]
        )
        seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask, stru_token)
        logits = self.classifier(seq_embeds, attention_mask)
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
