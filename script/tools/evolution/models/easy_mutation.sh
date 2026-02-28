#!/usr/bin/env bash
# Easy Mutation: ensemble structure-based prediction. Run from project root.

PYTHONPATH=src python -m tools.mutation.tools_cmd easy-mutation \
    --pdb_file download/alphafold2_structures/A0A1B0GTW7.pdb \
    --output_dir mutation/example/mycase/easy_mutation \
    --num_recommendations 30
