"""Tabular contracts for structure-property artifacts."""
def residue_rows(data, kind, chain="A"):
    if not isinstance(data, dict):
        raise ValueError("Structure-property output must contain residue records")
    groups = data.items() if kind == "sasa" else [(chain, data)]
    rows = []
    for chain_id, residues in groups:
        if not isinstance(residues, dict):
            raise ValueError("Invalid chain records in structure-property output")
        for residue, values in residues.items():
            if not isinstance(values, dict):
                raise ValueError("Invalid residue measurements")
            rows.append({"chain": str(chain_id), "residue": str(residue), **values})
    return rows
