#!/usr/bin/env bash
# NCBI BLAST: submit query and save result (requires Biopython).
# -i sequence or accession; -f FASTA file; -o output path; -p program; -d database.

# Example 1: query by accession (NCBI protein ID), blastp vs nr, save XML
python src/tools/search/database/ncbi/ncbi_blast.py \
    -i NP_000517.2 \
    -o example/database/ncbi/blast_result.xml \
    -p blastp \
    -d nr \
    -e 0.001

# Example 2: query from FASTA file, blastn vs nt
# python src/tools/search/database/ncbi/ncbi_blast.py \
#     -f example/database/ncbi/query.fasta \
#     -o example/database/ncbi/blast_nt.xml \
#     -p blastn \
#     -d nt

# Example 3: restrict to organism (e.g. mouse)
# python src/tools/search/database/ncbi/ncbi_blast.py \
#     -i NP_000517.2 \
#     -o example/database/ncbi/blast_mouse.xml \
#     -p blastp \
#     -d nr \
#     --entrez_query "Mus musculus[Organism]"
