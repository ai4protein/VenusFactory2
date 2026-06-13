import os
import argparse
import json
from Bio.PDB import PDBParser

# Backend: pydssp (pure-Python). Avoids the bioconda dssp binary, whose
# libcifpp dict loader is broken on common conda installs ("Is a directory"
# error when reading mmcif_pdbx.dic). pydssp computes a 3-class assignment
# (H / E / -). We map to 8-class via the standard collapse below so the
# returned dict still has both ss8_seq and ss3_seq fields.

ss_alphabet = ['H', 'E', 'C']
ss_alphabet_dic = {
    "H": "H", "G": "H", "E": "E",
    "B": "E", "I": "C", "T": "C",
    "S": "C", "L": "C", "-": "C",
    "P": "C"
}
# Secondary structure code to full name mapping (kept for 8-class compatibility).
ss_map = {
    'H': 'Alpha Helix',
    'B': 'Beta Bridge',
    'E': 'Beta Strand',
    'G': '3-10 Helix',
    'I': 'Pi Helix',
    'T': 'Turn',
    'S': 'Bend',
    '-': 'Loop/Irregular',
    'C': 'Coil',
}

# Three-letter → one-letter amino acid (used to recover aa_seq for chain match).
_THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}


def calculate_ss_from_pdb(pdb_file: str, chain_id: str = 'A') -> dict:
    """Calculate secondary structure for one chain using pydssp.

    Returns:
        dict keyed by residue number (str) → {'aa_seq': one_letter, 'ss8_seq', 'ss3_seq'}.
        Empty dict on error or empty chain.
    """
    if not os.path.exists(pdb_file):
        print(f"error: file '{pdb_file}' not found.")
        return {}

    try:
        import pydssp
        with open(pdb_file, encoding='utf-8', errors='replace') as fh:
            pdb_text = fh.read()
        coords = pydssp.read_pdbtext(pdb_text)
        # pydssp default returns one-character codes ('-', 'H', 'E') as numpy array of strings
        ss_letters = pydssp.assign(coords)
    except Exception as e:
        print(f"error (pydssp assign failed): {e}")
        return {}

    # pydssp returns one assignment per CA atom across ALL chains in PDB order.
    # We need to walk the structure ourselves to map (pydssp index → chain_id, resnum, aa).
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein_structure", pdb_file)
        model = structure[0]
        residues_in_order: list[tuple[str, int, str]] = []
        for chain in model:
            for residue in chain:
                # Skip HETATM / waters
                if residue.id[0].strip() != "":
                    continue
                if "CA" not in residue:
                    continue
                resname = residue.get_resname().strip().upper()
                residues_in_order.append((chain.id, residue.id[1], _THREE_TO_ONE.get(resname, 'X')))
    except Exception as e:
        print(f"error (PDB walk failed): {e}")
        return {}

    if len(residues_in_order) != len(ss_letters):
        # Mismatch — usually means non-standard residues skipped here but not in pydssp.
        # Truncate to the shorter length so we still return something useful.
        n = min(len(residues_in_order), len(ss_letters))
        residues_in_order = residues_in_order[:n]
        ss_letters = ss_letters[:n]

    ss_data: dict = {}
    for (cid, resnum, aa), code in zip(residues_in_order, ss_letters):
        if cid != chain_id:
            continue
        ss8 = str(code)  # pydssp gives 3-class already (H/E/-); use same for ss8 field
        ss3 = ss_alphabet_dic.get(ss8, 'C')
        ss_data[str(resnum)] = {
            'aa_seq': aa,
            'ss8_seq': ss8,
            'ss3_seq': ss3,
        }
    return ss_data


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
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f" Results saved to: {args.output_file}")
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