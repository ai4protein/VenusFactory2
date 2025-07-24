from dash import Dash, html, dcc, Input, Output, no_update, callback_context
import dash_bio as dashbio
from dash_bio.utils import PdbParser
import os
import time
import hashlib
from functools import lru_cache

CONFIG_FILE = "./src/web/pdb_config.txt"

app = Dash(__name__)

_file_cache = {}
_last_config_check = 0
_config_check_interval = 2

app.layout = html.Div([
    dcc.Store(id='current-pdb-path-store', data=''),
    dcc.Store(id='molecule-data-store', data=None),
    dcc.Interval(id='interval-checker', interval=_config_check_interval * 1000, n_intervals=0),
    html.Div(id='viewer-container')
])

def get_file_hash(file_path):
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

@lru_cache(maxsize=10)
def parse_plddt_from_pdb_cached(pdb_file_path, file_hash):
    plddt_scores = {}
    
    try:
        with open(pdb_file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain_id = line[21:22].strip()
                    residue_number = int(line[22:26].strip())
                    b_factor = float(line[60:66].strip())
                    
                    key = f"{chain_id}_{residue_number}"
                    plddt_scores[key] = b_factor
                    
    except Exception as e:
        print(f"Error parsing PDB: {e}")
        
    return plddt_scores

def get_plddt_color(plddt_score):
    if plddt_score > 90:
        return '#0053D6'  # Very high confidence
    elif plddt_score > 70:
        return '#65CBF3'  # Confident
    elif plddt_score > 50:
        return '#FFDB13'  # Low confidence
    else:
        return '#FF7D45'  # Very low confidence

def process_pdb_file(pdb_path):
    file_hash = get_file_hash(pdb_path)
    if not file_hash:
        return None
    
    cache_key = f"{pdb_path}_{file_hash}"
    if cache_key in _file_cache:
        return _file_cache[cache_key]
    
    
    try:
        parser = PdbParser(pdb_path)
        data = parser.mol3d_data()
        
        plddt_scores = parse_plddt_from_pdb_cached(pdb_path, file_hash)
        
        styles = []
        
        for i, atom in enumerate(data['atoms']):
            chain = atom.get('chain', '')
            residue_number = atom.get('residue_index', 0)
            residue_key = f"{chain}_{residue_number}"
            
            plddt = plddt_scores.get(residue_key, 50)
            color = get_plddt_color(plddt)
            
            atom_style = {
                'visualization_type': 'cartoon',
                'color': color,
                'residue_index': residue_number,
                'chain': chain,
            }
            
            styles.append(atom_style)
        
        result = {
            'data': data,
            'styles': styles,
            'file_path': pdb_path
        }
        
        _file_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        print(f"Error processing PDB file: {e}")
        return None

@app.callback(
    Output('molecule-data-store', 'data'),
    Output('current-pdb-path-store', 'data'),
    Input('interval-checker', 'n_intervals'),
    Input('current-pdb-path-store', 'data')
)
def check_for_new_file(n, current_path):
    global _last_config_check
    
    _last_config_check = current_time
    
    try:
        if not os.path.exists(CONFIG_FILE):
            return no_update, no_update
        
        with open(CONFIG_FILE, 'r') as f:
            new_path = f.read().strip()
            
        if not new_path or new_path == current_path:
            return no_update, no_update

        if not os.path.exists(new_path):
            return None, current_path
        
        processed_data = process_pdb_file(new_path)
        
        return processed_data, new_path
        
    except Exception as e:
        print(f"Error in file check: {e}")
        return no_update, no_update

@app.callback(
    Output('viewer-container', 'children'),
    Input('molecule-data-store', 'data')
)
def update_viewer(molecule_data):
    if not molecule_data:
        return html.Div([
            html.H4("Waiting for structure data...", style={'text-align': 'center', 'color': '#666'}),
            html.P("Download a structure to view it here.", style={'text-align': 'center', 'color': '#999'})
        ])
    
    try:
        return html.Div([
            html.H4(f"Viewing: {os.path.basename(molecule_data['file_path'])}", 
                   style={'text-align': 'center', 'margin-bottom': '10px'}),
            dashbio.Molecule3dViewer(
                id='molecule3d',
                modelData=molecule_data['data'],
                styles=molecule_data['styles'],
                backgroundColor='#FFFFFF',
                height=500,
                width=900,
            )
        ])
        
    except Exception as e:
        return html.Div(f"Error rendering molecule: {str(e)}", 
                       style={'color': 'red', 'text-align': 'center'})

def run_dash_server():
    app.run(
        debug=False,
        port=8050,
        host='127.0.0.1', 
        dev_tools_silence_routes_logging=True,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )

if __name__ == '__main__':
    run_dash_server()