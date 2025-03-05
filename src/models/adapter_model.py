import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .pooling import Attention1dPoolingHead, MeanPoolingHead, LightAttentionPoolingHead
from .pooling import MeanPooling, MeanPoolingProjection

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class CrossModalAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_head_size = args.hidden_size // args.num_attention_head
        assert (
            self.attention_head_size * args.num_attention_head == args.hidden_size
        ), "Embed size needs to be divisible by num heads"
        self.num_attention_head = args.num_attention_head
        self.hidden_size = args.hidden_size
        
        self.query_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.key_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.value_proj = nn.Linear(args.hidden_size, args.hidden_size)
        
        self.dropout = nn.Dropout(args.attention_probs_dropout)
        
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_head, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, attention_mask=None, output_attentions=False):
        key_layer = self.transpose_for_scores(self.key_proj(key))
        value_layer = self.transpose_for_scores(self.value_proj(value))
        query_layer = self.transpose_for_scores(self.query_proj(query))
        query_layer = query_layer * self.attention_head_size**-0.5
        
        query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else context_layer
        
        return outputs

class AdapterModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if 'foldseek_seq' in args.structure_seq:
            self.foldseek_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
            self.cross_attention_foldseek = CrossModalAttention(args)
        if 'ss8_seq' in args.structure_seq:
            self.ss_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
            self.cross_attention_ss = CrossModalAttention(args)
        if 'esm3_structure_seq' in args.structure_seq:
            self.esm3_structure_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
            self.cross_attention_esm3_structure = CrossModalAttention(args)
        
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        
        if args.pooling_method == 'attention1d':
            self.classifier = Attention1dPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
        elif args.pooling_method == 'mean':
            if "PPI" in args.dataset:
                self.pooling = MeanPooling()
                self.projection = MeanPoolingProjection(args.hidden_size, args.num_labels, args.pooling_dropout)
            else:
                self.classifier = MeanPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
        elif args.pooling_method == 'light_attention':
            self.classifier = LightAttentionPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
        else:
            raise ValueError(f"classifier method {args.pooling_method} not supported")
    
    def plm_embedding(self, plm_model, aa_seq, attention_mask, structure_tokens):
        if "ProSST" in self.args.plm_model:
            outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, ss_input_ids=structure_tokens, output_hidden_states=True)
        elif "Prime" in self.args.plm_model:
            outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, output_hidden_states=True)
        elif self.training and hasattr(self, 'args') and self.args.training_method == 'full':
            outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        if "ProSST" in self.args.plm_model or "Prime" in self.args.plm_model:
            seq_embeds = outputs.hidden_states[-1]
        else:
            seq_embeds = outputs.last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds
    
    def forward(self, plm_model, batch):
        aa_seq, attention_mask, stru_tokens = batch['aa_seq_input_ids'], batch['aa_seq_attention_mask'], batch['aa_seq_stru_tokens']
        seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask, stru_tokens)

        if 'foldseek_seq' in self.args.structure_seq:
            foldseek_seq = batch['foldseek_seq_input_ids']
            foldseek_embeds = self.foldseek_embedding(foldseek_seq)
            foldseek_embeds = self.cross_attention_foldseek(foldseek_embeds, seq_embeds, seq_embeds, attention_mask)
            embeds = seq_embeds + foldseek_embeds
            embeds = self.layer_norm(embeds)
        
        if 'ss8_seq' in self.args.structure_seq:
            ss_seq = batch['ss8_seq_input_ids']
            ss_embeds = self.ss_embedding(ss_seq)
            
            if 'foldseek_seq' in self.args.structure_seq:
                # cross attention with foldseek
                ss_embeds = self.cross_attention_ss(ss_embeds, embeds, embeds, attention_mask)
                embeds = ss_embeds + embeds
            else:
                # cross attention with sequence
                ss_embeds = self.cross_attention_ss(ss_embeds, seq_embeds, seq_embeds, attention_mask)
                embeds = ss_embeds + seq_embeds
            embeds = self.layer_norm(embeds)
        
        if 'esm3_structure_seq' in self.args.structure_seq:
            esm3_structure_seq = batch['esm3_structure_seq_input_ids']
            esm3_structure_embeds = self.esm3_structure_embedding(esm3_structure_seq)
            
            if 'foldseek_seq' in self.args.structure_seq:
                # cross attention with foldseek
                esm3_structure_embeds = self.cross_attention_esm3_structure(esm3_structure_embeds, embeds, embeds, attention_mask)
                embeds = esm3_structure_embeds + embeds
            elif 'ss8_seq' in self.args.structure_seq:
                # cross attention with ss8
                esm3_structure_embeds = self.cross_attention_esm3_structure(esm3_structure_embeds, ss_embeds, ss_embeds, attention_mask)
                embeds = esm3_structure_embeds + ss_embeds
            else:
                # cross attention with sequence
                esm3_structure_embeds = self.cross_attention_esm3_structure(esm3_structure_embeds, seq_embeds, seq_embeds, attention_mask)
                embeds = esm3_structure_embeds + seq_embeds
            embeds = self.layer_norm(embeds)
        
        if self.args.structure_seq:
            logits = self.classifier(embeds, attention_mask)
        else:
            logits = self.classifier(seq_embeds, attention_mask)            
        
        return logits
       