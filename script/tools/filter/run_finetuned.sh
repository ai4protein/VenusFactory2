#!/usr/bin/env bash
# Finetuned prediction (local via tools_cmd). Run from project root.
# Uses ckpt/<task>/<model_name>/ (e.g. ckpt/DeepET_Topt/ankh-large/lr5e-4_bt12k_ga8.pt).
# Supported --task:
#   DeepET_Topt, DeepLocBinary, DeepLocMulti, DeepSol, DeepSoluE,
#   DLKcat, EpHod, MetalIonBinding, ProtSolM, SortingSignal, Thermostability,
#   VenusVaccine_VirusBinary, VenusVaccine_BacteriaBinary, VenusVaccine_TumorBinary,
#   VenusX_Res_Act_MP90, VenusX_Res_BindI_MP90, VenusX_Res_Evo_MP90, VenusX_Res_Motif_MP90
# --model_name: backbone variant, e.g. ankh-large, ProtT5-xl-uniref50

python src/tools/predict/tools_cmd.py \
    finetuned \
    --task VenusX_Res_Act_MP90 \
    --model_name ESM2-650M \
    --fasta_file example/download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_csv example/predict/output/A0A0C5B5G6_esm2_venusx_res_act_mp90.csv
