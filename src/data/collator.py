import torch
import re
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer
from dataclasses import dataclass

VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = {
    "MASK": VQVAE_CODEBOOK_SIZE,
    "EOS": VQVAE_CODEBOOK_SIZE + 1,
    "BOS": VQVAE_CODEBOOK_SIZE + 2,
    "PAD": VQVAE_CODEBOOK_SIZE + 3,
    "CHAINBREAK": VQVAE_CODEBOOK_SIZE + 4,
}

@dataclass
class Collator:
    """Data collator class for protein sequences."""
    tokenizer: PreTrainedTokenizer
    max_length: int = None
    structure_seq: List[str] = None
    problem_type: str = 'classification'
    plm_model: str = None
    num_labels: int = None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching examples."""
        # Initialize lists to store sequences and labels
        aa_seqs, labels, str_tokens = [], [], []
        structure_seqs = {
            seq_type: [] for seq_type in (self.structure_seq or [])
        }

        # Process each example
        for e in examples:
            # Process sequences
            aa_seq = self.process_sequence(e["aa_seq"])
            aa_seqs.append(aa_seq)
            stru_token = self.process_stru_tokens(e["prosst_stru_token"])
            str_tokens.append(stru_token)
            
            # Process structure sequences if needed
            for seq_type in structure_seqs:
                if seq_type == 'esm3_structure_seq':
                    processed_seq = self.process_esm3_structure_seq(e[seq_type])
                else:
                    processed_seq = self.process_sequence(e[seq_type])
                structure_seqs[seq_type].append(processed_seq)
            
            if self.problem_type == 'multi_label_classification':
                label_list = e['label'].split(',')
                e['label'] = [int(l) for l in label_list]
                binary_list = [0] * self.num_labels
                for index in e['label']:
                    binary_list[index] = 1
                e['label'] = binary_list
            # Process labels
            labels.append(e["label"])

        # Tokenize sequences
        batch = self.tokenize_sequences(aa_seqs, structure_seqs, str_tokens)
        
        # Add labels to batch
        batch["label"] = torch.as_tensor(
            labels, 
            dtype=torch.float if self.problem_type == 'regression' else torch.long
        )

        return batch

    def process_sequence(self, seq: str) -> str:
        """Process sequence based on model type."""
        if 'prot_bert' in self.plm_model or "prot_t5" in self.plm_model:
            seq = " ".join(list(seq))
            seq = re.sub(r"[UZOB]", "X", seq)
        return seq

    def process_esm3_structure_seq(self, seq: List[int]) -> torch.Tensor:
        """Process ESM3 structure sequence."""
        return torch.tensor([VQVAE_SPECIAL_TOKENS["BOS"]] + seq + [VQVAE_SPECIAL_TOKENS["EOS"]])

    def process_stru_tokens(self, seq:List[int]) -> torch.Tensor:
        """Process ProSST structure token."""
        if isinstance(seq, str):
            seq_clean = seq.strip("[]").replace(" ","")
            tokens = list(map(int, seq_clean.split(','))) if seq_clean else []
        elif isinstance(seq, (list, tuple)):
            tokens = [int(x) for x in seq]
        stru_tokens = [int(num) for num in tokens]
        return torch.tensor(stru_tokens)
    
    def tokenize_sequences(
        self,
        aa_seqs: List[str],
        structure_seqs: Dict[str, List[str]],
        str_tokens: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize all sequences."""
        # Process amino acid sequences
        aa_encodings = self.tokenizer(
            aa_seqs,
            padding=True,
            truncation=True if self.max_length else False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        aa_max_length = len(aa_encodings["input_ids"][0])
        padded_tokens = []
        for tokens in str_tokens:
            struct_sequence =  [int(num) for num in tokens]
            padded_tokens.append(struct_sequence + [0] * (aa_max_length - len(struct_sequence) - 2))


        batch = {
            "aa_seq_input_ids": aa_encodings["input_ids"],
            "aa_seq_attention_mask": aa_encodings["attention_mask"],
            "aa_seq_stru_tokens": torch.tensor(padded_tokens, dtype=torch.long)
        }
        
        # Process structure sequences if provided
        for seq_type, seqs in structure_seqs.items():
            if not seqs:
                continue
                
            if seq_type == 'esm3_structure_seq':
                # ESM3 structure sequences are already tokenized
                structure_tokens = torch.stack(seqs)
                # Pad sequences to max length
                max_len = max(len(seq) for seq in seqs)
                padded_tokens = torch.zeros(len(seqs), max_len, dtype=torch.long)
                attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
                
                for i, seq in enumerate(seqs):
                    seq_len = len(seq)
                    padded_tokens[i, :seq_len] = seq
                    attention_mask[i, :seq_len] = 1
                    
                batch[f"{seq_type}_input_ids"] = padded_tokens
                batch[f"{seq_type}_attention_mask"] = attention_mask
                
            else:
                # Tokenize other structure sequences
                structure_encodings = self.tokenizer(
                    seqs,
                    padding=True,
                    truncation=True if self.max_length else False,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                batch[f"{seq_type}_input_ids"] = structure_encodings["input_ids"]
                batch[f"{seq_type}_attention_mask"] = structure_encodings["attention_mask"]
        
        return batch 