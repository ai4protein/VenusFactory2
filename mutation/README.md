## zero-shot mutation scores

Rank from **high to low** to select the best mutation for wet-lab experiments.

### ProSST (highly recommended if you use AlphaFold2 or AlphaFold3 structure)

```bash
python src/mutation/models/prosst.py \
    --pdb_file download/alphafold2_structures/A0A1B0GTW7.pdb \
    --output_csv mutation/example/A0A1B0GTW7_prosst.csv
```

```bibtex
@inproceedings{li2024prosst,
  title={Pro{SST}: Protein Language Modeling with Quantized Structure and Disentangled Attention},
  author={Mingchen Li and Yang Tan and Xinzhu Ma and Bozitao Zhong and Huiqun Yu and Ziyi Zhou and Wanli Ouyang and Bingxin Zhou and Pan Tan and Liang Hong},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=4Z7RZixpJQ}
}
```

### ProtSSN

### ESM2

```bash
python src/mutation/models/esm2.py \
    --fasta_file download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_csv mutation/example/A0A0C5B5G6_esm2.csv
```

### ESM1V

```bash
python src/mutation/models/esm1v.py \
    --fasta_file download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_csv mutation/example/A0A0C5B5G6_esm1v.csv
```

