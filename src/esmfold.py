import torch
import esm
import os
import gc
import argparse
import biotite.structure.io as bsio
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from transformers import AutoTokenizer, EsmForProteinFolding

from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

def read_fasta(file_path, key):
    return str(getattr(SeqIO.read(file_path, 'fasta'), key))

def read_multi_fasta(file_path):
    """
    params:
        file_path: path to a fasta file
    return:
        a dictionary of sequences
    """
    sequences = {}
    current_sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences[header] = current_sequence
                    current_sequence = ''
                header = line
            else:
                current_sequence += line
        if current_sequence:
            sequences[header] = current_sequence
    return sequences

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--fasta_file", type=str, default=None)
    parser.add_argument("--fasta_chunk_num", type=int, default=None)
    parser.add_argument("--fasta_chunk_id", type=int, default=None)
    parser.add_argument("--fasta_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--out_file", type=str, default="result.pdb")
    parser.add_argument("--out_info_file", type=str, default=None)
    parser.add_argument("--fold_chunk_size", type=int)
    args = parser.parse_args()
    
    # model = esm.pretrained.esmfold_v1()
    # model = model.eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

    model = model.cuda()
    # model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
    # Lower sizes will have lower memory requirements at the cost of increased speed.
    if args.fold_chunk_size is not None:
        model.trunk.set_chunk_size(args.fold_chunk_size)
    
    if args.fasta_file is not None:
        unfold_proteins = []
        seq_dict = read_multi_fasta(args.fasta_file)
        os.makedirs(args.out_dir, exist_ok=True)
        names, sequences = list(seq_dict.keys()), list(seq_dict.values())
        if args.fasta_chunk_num is not None:
            chunk_size = len(names) // args.fasta_chunk_num + 1
            start = args.fasta_chunk_id * chunk_size
            end = min((args.fasta_chunk_id + 1) * chunk_size, len(names))
            names, sequences = names[start:end], sequences[start:end]
        
        out_info_dict = {"name": [], "plddt": []}
        bar = tqdm(zip(names, sequences))
        for name, sequence in bar:
            bar.set_description(name)
            name = name[1:].split(" ")[0]
            out_file = os.path.join(args.out_dir, f"{name}.ef.pdb")
            if os.path.exists(out_file):
                out_info_dict["name"].append(name)
                struct = bsio.load_structure(out_file, extra_fields=["b_factor"])
                out_info_dict["plddt"].append(struct.b_factor.mean())
                continue
            
            # Multimer prediction can be done with chains separated by ':'
            try:
                tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
                with torch.no_grad():
                    output = model(tokenized_input)
            except Exception as e:
                print(e)
                print(f"Failed to predict {name}")
                unfold_proteins.append(name)
                continue
            gc.collect()
            pdb = convert_outputs_to_pdb(output)
            with open(out_file, "w") as f:
                f.write("\n".join(pdb))
                
            out_info_dict["name"].append(name)
            struct = bsio.load_structure(out_file, extra_fields=["b_factor"])
            out_info_dict["plddt"].append(struct.b_factor.mean())
        
        if args.out_info_file is not None:
            pd.DataFrame(out_info_dict).to_csv(args.out_info_file, index=False)
        
    if args.fasta_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        proteins = sorted(os.listdir(args.fasta_dir))
        bar = tqdm(proteins)
        for p in bar:
            name = p[:-6]
            bar.set_description(name)
            out_file = os.path.join(args.out_dir, f"{name}.ef.pdb")
            if os.path.exists(out_file):
                continue
            bar.set_description(p)
            sequence = read_fasta(os.path.join(args.fasta_dir, p), "seq")
            tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
            # Multimer prediction can be done with chains separated by ':'

            with torch.no_grad():
                output = model(tokenized_input)

            pdb = convert_outputs_to_pdb(output)
            with open(out_file, "w") as f:
                f.write("\n".join(pdb))

            struct = bsio.load_structure(out_file, extra_fields=["b_factor"])
            print(p, struct.b_factor.mean())
    elif args.sequence is not None:
        sequence = args.sequence
        # Multimer prediction can be done with chains separated by ':'

        with torch.no_grad():
            output = model.infer_pdb(sequence)

        with open(args.out_file, "w") as f:
            f.write(output)

        struct = bsio.load_structure(args.out_file, extra_fields=["b_factor"])
        print(struct.b_factor.mean())