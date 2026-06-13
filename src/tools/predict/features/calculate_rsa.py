import os
import argparse
import json
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

# Backend: BioPython Shrake-Rupley → SASA(residue) ÷ MaxASA(residue_type) → RSA.
# Avoids the external DSSP binary, whose bioconda packaging has a broken
# libcifpp dict loader on common installs.
#
# MaxASA values from Tien et al. 2013 "Maximum allowed solvent accessibility of
# residues in proteins" (PLoS ONE), in Å². Same numbers DSSP uses internally.
_MAX_ASA_TIEN_2013 = {
    'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0,
    'Q': 225.0, 'E': 223.0, 'G': 104.0, 'H': 224.0, 'I': 197.0,
    'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0,
    'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0,
}
_THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


def calculate_rsa_from_pdb(pdb_file: str, chain_id: str = 'A') -> dict:
    """Calculate per-residue RSA on one chain (Shrake-Rupley SASA ÷ MaxASA).

    Returns:
        dict keyed by residue number (str) → {'aa': one_letter, 'rsa': float}.
        Empty dict on error or empty chain.
    """
    if not os.path.exists(pdb_file):
        print(f"error: file '{pdb_file}' not found.")
        return {}

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein_structure", pdb_file)
        model = structure[0]
        sasa_calc = ShrakeRupley()
        sasa_calc.compute(model, level='R')  # attaches .sasa (Å²) to each residue
    except Exception as e:
        print(f"error: {e}")
        return {}

    try:
        chain = model[chain_id]
    except KeyError:
        return {}

    rsa_data: dict = {}
    for residue in chain:
        if residue.id[0].strip() != "":  # skip HETATM / waters
            continue
        resname = residue.get_resname().strip().upper()
        aa = _THREE_TO_ONE.get(resname)
        if aa is None or aa not in _MAX_ASA_TIEN_2013:
            continue
        sasa = float(getattr(residue, "sasa", 0.0) or 0.0)
        max_asa = _MAX_ASA_TIEN_2013[aa]
        rsa = round(sasa / max_asa, 4) if max_asa > 0 else 0.0
        rsa_data[str(residue.id[1])] = {
            'aa': aa,
            'rsa': rsa,
        }
    return rsa_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate RSA from PDB file')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to the PDB file')
    parser.add_argument('--chain_id', type=str, default='A', help='Chain ID to analyze (default is "A")')
    parser.add_argument('--output_file', type=str, default=None, help='(optional) path to the output JSON file.\nIf not provided, the results will be printed to the screen.')
    args = parser.parse_args()

    # calculate the RSA value of chain 'A'
    all_rsa_values = calculate_rsa_from_pdb(args.pdb_file, chain_id=args.chain_id)

    if all_rsa_values:
        # Prepare results for output
        exposed_count = sum(1 for res in all_rsa_values.values() if res['rsa'] >= 0.25)
        buried_count = len(all_rsa_values) - exposed_count
        
        # Add location information to each residue
        for res_id, res_data in all_rsa_values.items():
            res_data['location'] = 'exposed' if res_data['rsa'] >= 0.25 else 'buried'
        
        results = {
            'chain_id': args.chain_id,
            'pdb_file': args.pdb_file,
            'exposed_residues': exposed_count,
            'buried_residues': buried_count,
            'total_residues': len(all_rsa_values),
            'residue_rsa': all_rsa_values
        }
        
        if args.output_file:
            # Write results to JSON file
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f" Results saved to: {args.output_file}")
            except IOError as e:
                print(f"error: cannot write to file '{args.output_file}': {e}")
        else:
            # Print results to screen
            print(f"successfully calculate the RSA value of chain '{args.chain_id}' in file '{args.pdb_file}'")
            print("-" * 50)
            print(f"Exposed residues: {exposed_count}")
            print(f"Buried residues: {buried_count}")
            print(f"Total residues: {len(all_rsa_values)}")
            print("-" * 50)
            for res_id, data in all_rsa_values.items():
                aa = data['aa']
                rsa = data['rsa']
                location = "Exposed (surface)" if rsa >= 0.25 else "Buried (core)"
                print(f"  residue {res_id} ({aa}): RSA = {rsa:.3f}  ({location})")
            print("-" * 50)