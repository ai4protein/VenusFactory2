import os
import sys
from pathlib import Path

_REPO_ROOT = next((p for p in Path(__file__).absolute().parents if (p / "src").is_dir()), Path(__file__).absolute().parent)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import argparse
import datetime
import json
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# make sure to install biopython: pip install biopython

def calculate_physchem_from_sequence(sequence: str) -> dict:
    """Compute properties directly from one raw sequence or one FASTA record.

    Reject ambiguous residues rather than silently dropping or substituting them.
    This is a local CPU calculation; no structure, model download or network is needed.
    """
    text = str(sequence or '').strip()
    if len(text) > 200_000:
        raise ValueError('Provide one protein sequence of at most 100,000 residues.')
    sequence_id = 'sequence'
    if text.startswith('>'):
        lines = text.splitlines()
        if any(line.lstrip().startswith('>') for line in lines[1:]):
            raise ValueError('Provide one sequence at a time, not multiple FASTA records.')
        sequence_id = lines[0][1:].strip().split(' ', 1)[0] or 'sequence'
        text = '\n'.join(lines[1:])
    cleaned = ''.join(text.split()).upper()
    if cleaned.endswith('*'):
        cleaned = cleaned[:-1]
    if not cleaned or len(cleaned) > 100_000:
        raise ValueError('Provide one protein sequence of 1 to 100,000 residues.')
    invalid = sorted(set(cleaned) - set('ACDEFGHIKLMNPQRSTVWY'))
    if invalid:
        raise ValueError('Only the 20 standard amino acids are supported; unsupported characters: ' + ', '.join(invalid))
    return _sequence_properties(cleaned, sequence_id)


def _sequence_properties(cleaned_sequence: str, sequence_id: str) -> dict:
    analysed_seq = ProteinAnalysis(cleaned_sequence)
    helix, turn, sheet = analysed_seq.secondary_structure_fraction()
    return {
        'sequence_id': sequence_id,
        'sequence_length': len(cleaned_sequence),
        'molecular_weight': round(analysed_seq.molecular_weight(), 2),
        'theoretical_pI': round(analysed_seq.isoelectric_point(), 2),
        'aromaticity': round(analysed_seq.aromaticity(), 3),
        'instability_index': round(analysed_seq.instability_index(), 2),
        'gravy': round(analysed_seq.gravy(), 3),
        'secondary_structure_fraction': {'helix': round(helix, 3), 'turn': round(turn, 3), 'sheet': round(sheet, 3)},
        'amino_acid_composition': dict(analysed_seq.amino_acids_percent),
    }


def calculate_physchem_from_fasta(fasta_file: str) -> dict:
    """
    read FASTA file, calculate the physical and chemical properties of the first sequence.

    Args:
        fasta_file: path to the FASTA file.

    Returns:
        a dictionary, containing various physical and chemical properties.
        for example: {'sequence_length': 129, 'molecular_weight': 14305.2, ...}
        if the file does not exist or cannot be processed, return an empty dictionary.
    """
    if not os.path.exists(fasta_file):
        print(f"error: file '{fasta_file}' not found.")
        return {}

    try:
        # 1. read the first sequence in the FASTA file
        record = next(SeqIO.parse(fasta_file, "fasta"))
        sequence = str(record.seq)

        # make sure the sequence does not contain illegal characters (e.g. '*')
        # ProtParam can only handle the standard 20 amino acids
        cleaned_sequence = sequence.replace('*', '').replace('X', '').replace('U', 'C')

        return _sequence_properties(cleaned_sequence, record.id)

    except StopIteration:
        print(f"error: FASTA file '{fasta_file}' is empty or the format is incorrect.")
        return {}
    except Exception as e:
        print(f"error: {e}")
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate the physical and chemical properties of proteins from FASTA file')
    parser.add_argument('--pdb_file', type=str, help='path to the PDB file')
    parser.add_argument('--chain_id', type=str, default='A', help='ID of the chain to analyze (default is "A")')
    parser.add_argument('--fasta_file', type=str, help='path to the FASTA file')
    parser.add_argument('--output_file', type=str, default=None, help='(optional) path to the output JSON file.\nIf not provided, the results will be printed to the screen.')
    args = parser.parse_args()

    if args.pdb_file:
        from mcp_tools.models.factory.training.cli_utils.data_utils import extract_seq_from_pdb

        tmp_fasta_file = f"tmp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.fasta"
        sequence = extract_seq_from_pdb(args.pdb_file, chain=args.chain_id)
        with open(tmp_fasta_file, 'w', encoding='utf-8') as f:
            f.write(f">{args.pdb_file}\n{sequence}")
        args.fasta_file = tmp_fasta_file
    elif not os.path.exists(args.fasta_file):
        print(f"error: file '{args.fasta_file}' not found.")
        exit(1)

    properties = calculate_physchem_from_fasta(args.fasta_file)

    if properties:
        if args.output_file:
            # Write results to JSON file
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(properties, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {args.output_file}")
            except IOError as e:
                print(f"error: cannot write to file '{args.output_file}': {e}")
        else:
            # Print results to screen
            print(f"successfully analyze the sequence '{properties['sequence_id']}' in file '{args.fasta_file}'")
            print("-" * 50)
            print(f"sequence length: {properties['sequence_length']} aa")
            print(f"molecular weight: {properties['molecular_weight'] / 1000:.2f} kDa")
            print(f"theoretical pI: {properties['theoretical_pI']}")
            print(f"aromaticity: {properties['aromaticity']}")
            print(f"instability index: {properties['instability_index']}")
            if properties['instability_index'] > 40:
                print("  ⚠️  predicted as unstable protein")
            else:
                print("  ✅ predicted as stable protein")
            print(f"gravy: {properties['gravy']}")

            ssf = properties['secondary_structure_fraction']
            print(f"secondary structure prediction: Helix={ssf['helix']}, Turn={ssf['turn']}, Sheet={ssf['sheet']}")
            print("-" * 50)

    if args.pdb_file:
        os.remove(tmp_fasta_file)