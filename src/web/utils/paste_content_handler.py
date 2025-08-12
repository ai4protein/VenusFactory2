import os
import tempfile
import re
from pathlib import Path
from typing import Tuple, Optional
import time

def detect_content_type(content: str) -> str:
    content = content.strip()
    
    if content.startswith('>'):
        return 'fasta'
    
    pdb_indicators = ['ATOM', 'HETATM', 'TER', 'END', 'MODEL', 'ENDMDL', 'CRYST1', 'ORIGX1', 'SCALE1', 'HEADER', 'TITLE', 'REMARK', 'SEQRES', 'DBREF', 'COMPND', 'SOURCE', 'KEYWDS', 'EXPDTA', 'AUTHOR', 'REVDAT', 'JRNL', 'MASTER', 'CONECT']
    lines = content.split('\n')
    pdb_line_count = 0
    
    for line in lines[:20]:
        if any(indicator in line for indicator in pdb_indicators):
            pdb_line_count += 1
    
    if pdb_line_count >= 1:
        return 'pdb'
    
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    content_chars = set(content.upper().replace('\n', '').replace(' ', ''))
    
    if content_chars and content_chars.issubset(amino_acids) and len(content) > 10:
        return 'fasta'
    
    return 'unknown'

def create_fasta_content(sequence: str, header: str = ">Pasted_sequence") -> str:
    clean_sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    formatted_sequence = '\n'.join([clean_sequence[i:i+60] for i in range(0, len(clean_sequence), 60)])
    
    return f"{header}\n{formatted_sequence}"

def save_pasted_content(content: str, content_type: str) -> Tuple[str, str]:
    try:
        if content_type == 'fasta':
            filename = "temp_fasta.fasta"
        else:
            filename = "temp_pdb.pdb"
        
        file_path = Path(filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        status_msg = f"✅ Content saved as temporary {content_type.upper()} file: {filename}"
        return str(file_path), status_msg
        
    except Exception as e:
        return "", f"❌ Error saving file: {str(e)}"

def process_pasted_content(content: str) -> Tuple[str, str, str]:
    if not content or not content.strip():
        return "", "", "❌ Please enter content"
    
    content_type = detect_content_type(content)
    
    if content_type == 'unknown':
        return "", "", "❌ Cannot recognize content format, please ensure it's valid FASTA or PDB format"
    
    file_path, status_msg = save_pasted_content(content, content_type)
    
    if not file_path:
        return "", "", status_msg
    
    return file_path, content_type, status_msg

def cleanup_temp_files():
    temp_files = ["temp_fasta.fasta", "temp_pdb.pdb"]
    for filename in temp_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
