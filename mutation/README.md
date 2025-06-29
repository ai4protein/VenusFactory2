# Zero-shot mutation scores

Rank from **high to low** to select the best mutation for wet-lab experiments.

## Sequence-structure models

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

### ProtSSN (highly recommended if you use RCSB PDB)

```bash
python src/mutation/models/protssn.py \
    --pdb_file download/alphafold2_structures/A0A0C5B5G6.pdb \
    --output_csv mutation/example/A0A0C5B5G6_protssn.csv
```

```bibtex
@article{tan2025protssn,
  article_type = {journal},
  title = {Semantical and geometrical protein encoding toward enhanced bioactivity and thermostability},
  author = {Tan, Yang and Zhou, Bingxin and Zheng, Lirong and Fan, Guisheng and Hong, Liang},
  volume = 13,
  year = 2025,
  month = {may},
  pub_date = {2025-05-02},
  pages = {RP98033},
  citation = {eLife 2025;13:RP98033},
  doi = {10.7554/eLife.98033},
  url = {https://doi.org/10.7554/eLife.98033},
  journal = {eLife},
  issn = {2050-084X},
  publisher = {eLife Sciences Publications, Ltd},
}
```

## Sequence-only models

### ESM2

```bash
python src/mutation/models/esm2.py \
    --fasta_file download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_csv mutation/example/A0A0C5B5G6_esm2.csv
```

```bibtex
@article{lin2023esm2,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

### ESM1V

```bash
python src/mutation/models/esm1v.py \
    --fasta_file download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_csv mutation/example/A0A0C5B5G6_esm1v.csv
```

```bibtex
@article{meier2021esm1v,
  title={Language models enable zero-shot prediction of the effects of mutations on protein function},
  author={Meier, Joshua and Rao, Roshan and Verkuil, Robert and Liu, Jason and Sercu, Tom and Rives, Alex},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={29287--29303},
  year={2021}
}
```

