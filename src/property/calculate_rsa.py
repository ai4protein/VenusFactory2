import os
import argparse
import json
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

# conda install -c ostrokach dssp
def calculate_rsa_from_pdb(pdb_file: str, chain_id: str = 'A') -> dict:
    """
    read pdb file, use DSSP to calculate the relative solvent accessible surface area (RSA) of each residue on the specified chain.

    Args:
        pdb_file: path to the pdb file.
        chain_id: the id of the chain to analyze (default is 'A').

    Returns:
        a dictionary, the key is the residue number (str), the value is a dictionary containing the amino acid name and RSA value.
        for example: {'10': {'aa': 'CYS', 'rsa': 0.45}, ...}
        if the file does not exist or cannot be processed, return an empty dictionary.
    """
    if not os.path.exists(pdb_file):
        print(f"error: file '{pdb_file}' not found.")
        return {}

    try:
        # 1. initialize the parser and load the structure
        parser = PDBParser()
        structure = parser.get_structure("protein_structure", pdb_file)
        model = structure[0] # usually only process the first model

        # 2. run DSSP
        # if your dssp program is not in the system path, you need to specify its path here, e.g., dssp=DSSP(model, pdb_file, dssp='/path/to/your/mkdssp')
        dssp = DSSP(model, pdb_file)

        # 3. extract RSA information
        rsa_data = {}
        for key in dssp.keys():
            # the key format of DSSP is (chain_id, residue_id_tuple)
            # the value format is (dssp_index, amino_acid, secondary_structure, relative_asa, ...)
            if key[0] == chain_id:
                residue_id = str(key[1][1])
                amino_acid_code = dssp[key][1]
                rsa_value = dssp[key][3] 
                
                rsa_data[residue_id] = {
                    'aa': amino_acid_code,
                    'rsa': rsa_value
                }
        
        return rsa_data

    except Exception as e:
        print(f"error: {e}")
        return {}


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