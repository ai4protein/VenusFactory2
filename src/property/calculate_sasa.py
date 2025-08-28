
import os
import argparse
import csv
import json
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

def calculate_sasa_from_pdb(pdb_file: str) -> dict:
    """
    calculate the SASA value of each residue in the PDB file using BioPython's SASA module (Shrake-Rupley algorithm).

    Args:
        pdb_file: path to the PDB file.

    Returns:
        a nested dictionary, organized by chain ID and residue number, containing SASA data.
        e.g.:
        {
            'A': {
                '10': {'resname': 'GLU', 'sasa': 150.45},
                ...
            },
            'B': { ... }
        }
        if the file does not exist or cannot be processed, return an empty dictionary.
    """
    if not os.path.exists(pdb_file):
        print(f"error: file '{pdb_file}' not found.")
        return {}
    
    try:
        # 1. initialize the PDB parser and load the structure
        #    QUIET=True can suppress the warning messages during loading
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein_structure", pdb_file)
        
        # usually we only process the first model
        model = structure[0]

        # 2. initialize the SASA calculator
        sasa_calculator = ShrakeRupley()
        
        # 3. run the calculation
        #    level="R" means we want to calculate SASA at the residue (Residue) level
        #    this function will directly attach the .sasa attribute to the structure object, instead of returning a new object
        sasa_calculator.compute(model, level="R")
        
        sasa_data = {}
        # 4. traverse the structure to extract the calculation results
        for chain in model:
            chain_id = chain.get_id()
            chain_data = sasa_data.setdefault(chain_id, {})
            
            for residue in chain:
                # filter out water molecules, ligands, etc. non-standard residues (HETATM)
                # the first element of the id tuple of standard amino acids is ' '
                if residue.get_id()[0] == ' ':
                    # ensure the .sasa attribute exists (calculation successful)
                    if hasattr(residue, 'sasa'):
                        # the second element of the id tuple of PDB residues is the number
                        res_num = str(residue.get_id()[1])
                        # the third element of the id tuple of PDB residues is the insertion code, here we merge it with the number
                        insert_code = residue.get_id()[2]
                        if insert_code != ' ':
                            res_num += insert_code
                        
                        chain_data[res_num] = {
                            'resname': residue.get_resname(),
                            'sasa': round(residue.sasa, 2)
                        }
        
        return sasa_data

    except Exception as e:
        print(f"error: cannot process file '{pdb_file}': {e}")
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='calculate the solvent accessible surface area (SASA) of each amino acid in the PDB file using BioPython (Shrake-Rupley algorithm).',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--pdb_file', type=str, required=True, help='path to the PDB file.')
    parser.add_argument('--chain_id', type=str, default=None, help='ID of the chain to analyze (e.g. "A").\nIf not provided, all chains will be analyzed.')
    parser.add_argument('--output_file', type=str, default=None, help='(optional) path to the output JSON file.\nIf not provided, the results will be printed to the screen.')
    
    args = parser.parse_args()

    all_sasa_data = calculate_sasa_from_pdb(args.pdb_file)

    if all_sasa_data:
        if args.output_file:
            # --- write the results to a JSON file ---
            try:
                # Prepare results for JSON output
                results = {
                    'pdb_file': args.pdb_file,
                    'algorithm': 'Shrake-Rupley',
                    'units': 'Å^2',
                    'chains': {}
                }
                
                for chain, residues in sorted(all_sasa_data.items()):
                    if args.chain_id and chain != args.chain_id:
                        continue
                    
                    # Calculate total SASA for this chain
                    total_sasa = sum(res['sasa'] for res in residues.values())
                    
                    results['chains'][chain] = {
                        'total_sasa': round(total_sasa, 2),
                        'residue_count': len(residues),
                        'residue_sasa': residues
                    }
                
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"\n✅ Results saved to: {args.output_file}")

            except IOError as e:
                print(f"error: cannot write to file '{args.output_file}': {e}")

        else:
            # --- print the results to the screen ---
            print(f"\nSASA calculation results for file '{args.pdb_file}' (units: Å^2, algorithm: Shrake-Rupley):")
            print("-" * 50)
            print(f"{'Chain':<6} {'Residue':<12} {'SASA (Å^2)':<15}")
            print("-" * 50)
            
            for chain, residues in sorted(all_sasa_data.items()):
                if args.chain_id and chain != args.chain_id:
                    continue
                
                # Calculate and display total SASA for this chain
                total_sasa = sum(res['sasa'] for res in residues.values())
                print(f"--- Chain {chain} (Total SASA: {total_sasa:.2f} Å^2) ---")

                try:
                    sorted_res_keys = sorted(residues.keys(), key=int)
                except ValueError:
                    sorted_res_keys = sorted(residues.keys())

                for res_num in sorted_res_keys:
                    data = residues[res_num]
                    res_id_str = f"{data['resname']}{res_num}"
                    print(f"{chain:<6} {res_id_str:<12} {data['sasa']:<15.2f}")
            print("-" * 50)