import csv
import os
from Bio.PDB import PDBParser
from Bio.SeqIO import PdbIO
from Bio.PDB.Polypeptide import protein_letters_3to1

def generate_mutations_from_sequence(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    mutations = []
    for i, original in enumerate(sequence):
        for mutant in amino_acids:
            if mutant != original: 
                mutation = f"{original}{i+1}{mutant}" 
                mutations.append(mutation) 
    return mutations

def generate_point_mutations(fasta_file, output_csv):
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    mutations = []
    for i, original in enumerate(sequence):
        for mutant in amino_acids:
            if mutant != original: 
                mutation = f"{original}{i+1}{mutant}" 
                mutations.append((mutation, 0)) 

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['mutant', 'DMS_score']) 
        for mutation, score in mutations:
            csv_writer.writerow([mutation, score]) 

