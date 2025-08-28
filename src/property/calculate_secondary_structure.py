import os
import argparse
import json
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

# make sure to install dssp: conda install -c salilab dssp
# make sure to install biopython: pip install biopython

ss_alphabet = ['H', 'E', 'C']
ss_alphabet_dic = {
    "H": "H", "G": "H", "E": "E",
    "B": "E", "I": "C", "T": "C",
    "S": "C", "L": "C", "-": "C",
    "P": "C"
}
# DSSP secondary structure code to full name mapping
ss_map = {
    'H': 'Alpha Helix',
    'B': 'Beta Bridge',
    'E': 'Beta Strand',
    'G': '3-10 Helix',
    'I': 'Pi Helix',
    'T': 'Turn',
    'S': 'Bend',
    '-': 'Loop/Irregular'
}

def calculate_ss_from_pdb(pdb_file: str, chain_id: str = 'A') -> dict:
    """
    read PDB file, use DSSP to calculate the secondary structure of each residue on the specified chain.

    Args:
        pdb_file: path to the PDB file.
        chain_id: ID of the chain to analyze (default is 'A').

    Returns:
        a dictionary, key is the residue number (str), value is a dictionary containing the amino acid name and secondary structure information.
        for example: {'10': {'aa': 'CYS', 'ss': 'Beta Strand'}, ...}
        if the file does not exist or cannot be processed, return an empty dictionary.
    """
    if not os.path.exists(pdb_file):
        print(f"error: file '{pdb_file}' not found.")
        return {}

    try:
        parser = PDBParser()
        structure = parser.get_structure("protein_structure", pdb_file)
        model = structure[0]
        
        # run DSSP. if your dssp program is not in the system path, you need to specify its path here.
        # for example: dssp=DSSP(model, pdb_file, dssp='/path/to/your/mkdssp')
        dssp = DSSP(model, pdb_file)

        ss_data = {}
        for key in dssp.keys():
            # the key format of DSSP is (chain_id, residue_id_tuple)
            # the value format of DSSP is (dssp_index, amino_acid, secondary_structure, rsa, ...)
            if key[0] == chain_id:
                residue_id = str(key[1][1])
                amino_acid_code = dssp[key][1]
                ss_code = dssp[key][2]
                
                ss_data[residue_id] = {
                    'aa_seq': amino_acid_code,
                    'ss8_seq': ss_code,
                    'ss3_seq': ss_alphabet_dic.get(ss_code, 'C')
                }
        
        return ss_data

    except Exception as e:
        print(f"error: {e}")
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calculate the secondary structure from PDB file')
    parser.add_argument('--pdb_file', type=str, required=True, help='path to the PDB file')
    parser.add_argument('--chain_id', type=str, default='A', help='ID of the chain to analyze (default is "A")')
    parser.add_argument('--output_file', type=str, default=None, help='(optional) path to the output JSON file.\nIf not provided, the results will be printed to the screen.')
    args = parser.parse_args()

    all_ss_values = calculate_ss_from_pdb(args.pdb_file, chain_id=args.chain_id)

    if all_ss_values:
        # Prepare sequences and counts
        aa_seq = ""
        ss8_seq = ""
        ss3_seq = ""
        
        for res_id, data in all_ss_values.items():
            aa_seq += data['aa_seq']
            ss8_seq += data['ss8_seq']
            ss3_seq += data['ss3_seq']
        
        # Count secondary structure elements
        ss_counts = {
            'helix': ss3_seq.count('H'),
            'sheet': ss3_seq.count('E'),
            'coil': ss3_seq.count('C')
        }
        
        # Add full names for secondary structure
        for res_id, res_data in all_ss_values.items():
            res_data['ss8_name'] = ss_map.get(res_data['ss8_seq'], 'Unknown')
        
        results = {
            'chain_id': args.chain_id,
            'pdb_file': args.pdb_file,
            'aa_sequence': aa_seq,
            'ss8_sequence': ss8_seq,
            'ss3_sequence': ss3_seq,
            'ss_counts': ss_counts,
            'residue_ss': all_ss_values
        }
        
        if args.output_file:
            # Write results to JSON file
            try:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Results saved to: {args.output_file}")
            except IOError as e:
                print(f"error: cannot write to file '{args.output_file}': {e}")
        else:
            # Print results to screen
            print(f"successfully calculate the secondary structure of chain '{args.chain_id}' in file '{args.pdb_file}'")
            print("-" * 50)
            print(f"Sequence length: {len(aa_seq)}")
            print(f"Helix (H): {ss_counts['helix']} ({ss_counts['helix']/len(aa_seq)*100:.1f}%)")
            print(f"Sheet (E): {ss_counts['sheet']} ({ss_counts['sheet']/len(aa_seq)*100:.1f}%)")
            print(f"Coil (C): {ss_counts['coil']} ({ss_counts['coil']/len(aa_seq)*100:.1f}%)")
            print("-" * 50)
            for res_id, data in all_ss_values.items():
                print(f"  residue {res_id} ({data['aa_seq']}): ss8: {data['ss8_seq']} ({data['ss8_name']}), ss3: {data['ss3_seq']}")
            print("-" * 50)
            print(f"aa_seq: {aa_seq}")
            print(f"ss8_seq: {ss8_seq}")
            print(f"ss3_seq: {ss3_seq}")
    else:
        print(f"error: cannot calculate the secondary structure of chain '{args.chain_id}' in file '{args.pdb_file}'")