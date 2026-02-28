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

    def plm_embedding(self, plm_model, aa_seq, attention_mask, stru_token=None):
        if (
            self.training
            and hasattr(self, "args")
            and self.args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']
        ):
            if "ProSST" in self.args.plm_model:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, ss_input_ids=stru_token, output_hidden_states=True)
            elif "Prime" in self.args.plm_model:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, output_hidden_states=True)
            else:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                if "ProSST" in self.args.plm_model:
                    outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, ss_input_ids=stru_token, output_hidden_states=True)
                else:
                    outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        if "ProSST" in self.args.plm_model:
            seq_embeds = outputs.hidden_states[-1]
        elif "Prime" in self.args.plm_model:
            seq_embeds = outputs.sequence_hidden_states[-1]
        else:
            seq_embeds = outputs.last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds

    def forward(self, plm_model, batch):
        if "ProSST" in self.args.plm_model:
            aa_seq, attention_mask, stru_token = (
                batch["aa_seq_input_ids"],
                batch["aa_seq_attention_mask"],
                batch["aa_seq_stru_tokens"]
            )
            seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask, stru_token)
        else:
            aa_seq, attention_mask = (
                batch["aa_seq_input_ids"],
                batch["aa_seq_attention_mask"],
            )            
            seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask)
        logits = self.classifier(seq_embeds, attention_mask)
        return logits
