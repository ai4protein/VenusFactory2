import os
import sys
sys.path.append(os.getcwd())
import json
import argparse
import datetime
import tempfile
from typing import Dict, Any
from Bio import SeqIO

# Import existing calculation functions
from src.property.calculate_physchem import calculate_physchem_from_fasta
from src.property.calculate_sasa import calculate_sasa_from_pdb
from src.property.calculate_rsa import calculate_rsa_from_pdb
from src.property.calculate_secondary_structure import calculate_ss_from_pdb
from src.utils.data_utils import extract_seq_from_pdb

def calculate_all_properties(input_file: str, file_type: str = 'auto', chain_id: str = 'A') -> Dict[str, Any]:
    """
    Calculate all available properties from input file by calling existing functions.
    
    Args:
        input_file: Path to input file (FASTA or PDB)
        file_type: Type of input file ('fasta', 'pdb', or 'auto')
        chain_id: Chain ID for PDB analysis
        
    Returns:
        Dictionary containing all calculated properties
    """
    # Auto-detect file type if not specified
    if file_type == 'auto':
        if input_file.lower().endswith('.fasta') or input_file.lower().endswith('.fa'):
            file_type = 'fasta'
        elif input_file.lower().endswith('.pdb'):
            file_type = 'pdb'
        else:
            raise ValueError("Cannot auto-detect file type. Please specify 'fasta' or 'pdb'.")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    
    # Initialize results dictionary
    results = {
        'input_file': input_file,
        'file_type': file_type,
        'chain_id': chain_id if file_type == 'pdb' else None,
        'analysis_timestamp': datetime.datetime.now().isoformat(),
        'properties': {}
    }
    
    try:
        if file_type == 'fasta':
            # Calculate physicochemical properties from FASTA
            physchem = calculate_physchem_from_fasta(input_file)
            if physchem:
                results['properties']['physicochemical'] = physchem
                print(f"✓ Calculated physicochemical properties for sequence '{physchem['sequence_id']}'")
            
        elif file_type == 'pdb':
            # Create temporary FASTA file for physicochemical calculation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_fasta:
                sequence = extract_seq_from_pdb(input_file, chain=chain_id)
                tmp_fasta.write(f">{os.path.basename(input_file)}_chain_{chain_id}\n{sequence}\n")
                tmp_fasta_path = tmp_fasta.name
            
            try:
                # Calculate physicochemical properties
                physchem = calculate_physchem_from_fasta(tmp_fasta_path)
                if physchem:
                    results['properties']['physicochemical'] = physchem
                    print(f"✓ Calculated physicochemical properties for chain {chain_id}")
                
                # Calculate SASA properties
                sasa = calculate_sasa_from_pdb(input_file)
                if sasa and chain_id in sasa:
                    # Convert to standardized format
                    sasa_data = sasa[chain_id]
                    total_sasa = sum(res['sasa'] for res in sasa_data.values())
                    results['properties']['sasa'] = {
                        'chain_id': chain_id,
                        'total_sasa': round(total_sasa, 2),
                        'residue_count': len(sasa_data),
                        'residue_sasa': sasa_data
                    }
                    print(f"✓ Calculated SASA properties for chain {chain_id}")
                
                # Calculate RSA properties
                rsa = calculate_rsa_from_pdb(input_file, chain_id)
                if rsa:
                    # Convert to standardized format
                    exposed_count = sum(1 for res in rsa.values() if res['rsa'] >= 0.25)
                    buried_count = len(rsa) - exposed_count
                    
                    # Add location information
                    for res_id, res_data in rsa.items():
                        res_data['location'] = 'exposed' if res_data['rsa'] >= 0.25 else 'buried'
                    
                    results['properties']['rsa'] = {
                        'chain_id': chain_id,
                        'exposed_residues': exposed_count,
                        'buried_residues': buried_count,
                        'residue_rsa': rsa
                    }
                    print(f"✓ Calculated RSA properties for chain {chain_id}")
                
                # Calculate secondary structure properties
                ss = calculate_ss_from_pdb(input_file, chain_id)
                if ss:
                    # Convert to standardized format
                    aa_seq = ''.join(res['aa_seq'] for res in ss.values())
                    ss8_seq = ''.join(res['ss8_seq'] for res in ss.values())
                    ss3_seq = ''.join(res['ss3_seq'] for res in ss.values())
                    
                    ss_counts = {
                        'helix': ss3_seq.count('H'),
                        'sheet': ss3_seq.count('E'),
                        'coil': ss3_seq.count('C')
                    }
                    
                    # Add full names for secondary structure
                    ss_map = {
                        'H': 'Alpha Helix', 'B': 'Beta Bridge', 'E': 'Beta Strand',
                        'G': '3-10 Helix', 'I': 'Pi Helix', 'T': 'Turn',
                        'S': 'Bend', '-': 'Loop/Irregular'
                    }
                    
                    for res_id, res_data in ss.items():
                        res_data['ss8_name'] = ss_map.get(res_data['ss8_seq'], 'Unknown')
                    
                    results['properties']['secondary_structure'] = {
                        'chain_id': chain_id,
                        'aa_sequence': aa_seq,
                        'ss8_sequence': ss8_seq,
                        'ss3_sequence': ss3_seq,
                        'ss_counts': ss_counts,
                        'residue_ss': ss
                    }
                    print(f"✓ Calculated secondary structure properties for chain {chain_id}")
                    
            finally:
                # Clean up temporary file
                os.unlink(tmp_fasta_path)
        
        return results
        
    except Exception as e:
        print(f"Error calculating properties: {e}")
        return results

def print_summary(results: Dict[str, Any]):
    """
    Print a summary of calculated properties.
    
    Args:
        results: Results dictionary from calculate_all_properties
    """
    print("\n" + "="*60)
    print("PROTEIN PROPERTY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Input file: {results['input_file']}")
    print(f"File type: {results['file_type']}")
    if results['chain_id']:
        print(f"Chain ID: {results['chain_id']}")
    print(f"Analysis time: {results['analysis_timestamp']}")
    
    properties = results['properties']
    
    if 'physicochemical' in properties:
        phys = properties['physicochemical']
        print(f"\n📊 PHYSICOCHEMICAL PROPERTIES:")
        print(f"  Sequence ID: {phys['sequence_id']}")
        print(f"  Length: {phys['sequence_length']} amino acids")
        print(f"  Molecular weight: {phys['molecular_weight']/1000:.2f} kDa")
        print(f"  Theoretical pI: {phys['theoretical_pI']}")
        print(f"  Aromaticity: {phys['aromaticity']}")
        print(f"  Instability index: {phys['instability_index']}")
        print(f"  GRAVY: {phys['gravy']}")
        
        ssf = phys['secondary_structure_fraction']
        print(f"  Secondary structure prediction: Helix={ssf['helix']}, Turn={ssf['turn']}, Sheet={ssf['sheet']}")
        
        if phys['instability_index'] > 40:
            print("  ⚠️  Predicted as unstable protein")
        else:
            print("  ✅ Predicted as stable protein")
    
    if 'sasa' in properties:
        sasa = properties['sasa']
        print(f"\n🌊 SASA PROPERTIES (Chain {sasa['chain_id']}):")
        print(f"  Total SASA: {sasa['total_sasa']} Å²")
        print(f"  Residues analyzed: {len(sasa['residue_sasa'])}")
    
    if 'rsa' in properties:
        rsa = properties['rsa']
        print(f"\n🔍 RSA PROPERTIES (Chain {rsa['chain_id']}):")
        print(f"  Exposed residues: {rsa['exposed_residues']}")
        print(f"  Buried residues: {rsa['buried_residues']}")
        print(f"  Total residues: {rsa['exposed_residues'] + rsa['buried_residues']}")
    
    if 'secondary_structure' in properties:
        ss = properties['secondary_structure']
        print(f"\n🧬 SECONDARY STRUCTURE (Chain {ss['chain_id']}):")
        print(f"  Sequence length: {len(ss['aa_sequence'])}")
        counts = ss['ss_counts']
        total_len = len(ss['aa_sequence'])
        if total_len > 0:
            print(f"  Helix (H): {counts['helix']} ({counts['helix']/total_len*100:.1f}%)")
            print(f"  Sheet (E): {counts['sheet']} ({counts['sheet']/total_len*100:.1f}%)")
            print(f"  Coil (C): {counts['coil']} ({counts['coil']/total_len*100:.1f}%)")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Calculate comprehensive protein properties from FASTA or PDB files using existing calculation modules',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_file', type=str, required=True, 
                       help='Path to input file (FASTA or PDB)')
    parser.add_argument('--file_type', type=str, choices=['fasta', 'pdb', 'auto'], default='auto',
                       help='Type of input file (default: auto-detect)')
    parser.add_argument('--chain_id', type=str, default='A',
                       help='Chain ID for PDB analysis (default: A)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output JSON file path (default: print to stdout)')
    parser.add_argument('--print_summary', action='store_true',
                       help='Print human-readable summary to stdout')
    
    args = parser.parse_args()
    
    try:
        # Calculate all properties
        results = calculate_all_properties(args.input_file, args.file_type, args.chain_id)
        
        # Print summary if requested
        if args.print_summary:
            print_summary(results)
        
        # Output results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved to: {args.output_file}")
        else:
            # Print JSON to stdout
            print(json.dumps(results, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
