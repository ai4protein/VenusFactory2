import numpy as np
import os
import json
from tqdm import tqdm

pdbs = os.listdir('alphafold_pdb')
for pdb in tqdm(pdbs):
    pdb_lines = open(f"alphafold_pdb/{pdb}").read().splitlines()

    def add_noise_and_save(variance, file_name):
        with open(file_name, "w") as file:
            for line in pdb_lines:
                if line.startswith("ATOM"):
                    parts = line.split()
                    try:
                        coords = np.array([float(parts[6]), float(parts[7]), float(parts[8])])
                        noise = np.random.normal(0, variance, coords.shape)
                        new_coords = coords + noise
                        new_line = f"{line[:30]}{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}{line[54:]}"
                        file.write(new_line + "\n")
                    except:
                        file.write(line + "\n")
                else:
                    file.write(line + "\n")

    variances = [0.5]

    for variance in variances:
        file_name = f"alphafold_pdb_noise_{variance}/{pdb}"
        try:
            add_noise_and_save(variance, file_name)
        except Exception as e:
            print(e) 
            print(pdb)
            
