"""Validate LigandMPNN design masks against the uploaded protein backbone."""
import math

def validate_ligand_design(text, parameters):
    sites = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        atom = line[12:16].strip()
        if atom not in {"N", "CA", "C", "O"}:
            continue
        try:
            occupancy = float(line[54:60])
            coordinates = [float(line[start:start+8]) for start in (30, 38, 46)]
            number = str(int(line[22:26]))
        except ValueError:
            raise ValueError("The PDB contains invalid backbone coordinates") from None
        if not all(math.isfinite(v) for v in [occupancy, *coordinates]):
            raise ValueError("The PDB contains non-finite backbone coordinates")
        if occupancy <= 0 and not parameters["parse_atoms_with_zero_occupancy"]:
            continue
        label = line[21] + number + line[26].strip()
        sites.setdefault(label, set()).add(atom)
    complete = {label for label, atoms in sites.items() if atoms == {"N", "CA", "C", "O"}}
    chains = {label[0] for label in complete}
    requested = set(parameters["chains_to_design"].split(",")) if parameters["chains_to_design"] else chains
    if requested - chains:
        raise ValueError("Choose a protein chain with complete backbone atoms; available: " + ", ".join(sorted(chains)))
    parsed = set(parameters["parse_these_chains_only"].split(",")) if parameters["parse_these_chains_only"] else chains
    selected = {label for label in complete if label[0] in requested & parsed}
    fixed = set(parameters["fixed_residues"].split())
    redesigned = set(parameters["redesigned_residues"].split())
    if (fixed | redesigned) - complete:
        raise ValueError("Design residue labels must match complete PDB backbone residues")
    selected = selected & redesigned if redesigned else selected - fixed
    if not selected:
        raise ValueError("The chain and residue settings leave no protein residues to design")
    return len(selected)
