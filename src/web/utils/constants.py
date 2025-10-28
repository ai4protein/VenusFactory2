"""Constants and configuration mappings for Venus Factory."""

# Model mappings for zero-shot prediction
MODEL_MAPPING_ZERO_SHOT = {
    "ESM2-650M": "esm2", 
    "ESM-1b": "esm1b",
    "ESM-IF1": "esmif1", 
    "ESM-1v": "esm1v",
    "SaProt": "saprot", 
    "MIF-ST": "mifst", 
    "ProSST-2048": "prosst", 
    "ProtSSN": "protssn",
    "VenusPLM": "venusplm"
}

DATASET_MAPPING_ZERO_SHOT = [
    "Activity",
    "Binding",
    "Expression",
    "Organismal Fitness",
    "Stability"
]

MODEL_MAPPING_FUNCTION = {
    "ESM2-650M": "esm2", 
    "Ankh-large": "ankh",
    "ProtBert": "protbert", 
    "ProtT5-xl-uniref50": "prott5",
}

MODEL_ADAPTER_MAPPING_FUNCTION = {
    "esm2": "esm2_t33_650M_UR50D", 
    "ankh": "ankh-large",
    "protbert": "prot_bert", 
    "prott5": "prot_t5_xl_uniref50",
}

MODEL_RESIDUE_MAPPING_FUNCTION = {
    "ESM2-650M": "esm2",
    "Ankh-large": "ankh",
    "ProtT5-xl-uniref50": "prott5",
}

DATASET_MAPPING_FUNCTION = {
    "Solubility": ["DeepSol", "DeepSoluE", "ProtSolM"],
    "Localization": ["DeepLocMulti"],
    "Membrane Protein Identification": ["DeepLocBinary"],
    "Metal ion binding": ["MetalIonBinding"], 
    "Stability": ["Thermostability"],
    "Sortingsignal": ["SortingSignal"], 
    "Optimum temperature": ["DeepET_Topt"],
    "Kcat": ["DLKcat"],
    "PH": ["EpHod"],
    "Virus": ["VenusVaccine_VirusBinary"],
    "Bacteria": ["VenusVaccine_BacteriaBinary"],
    "Tumor": ["VenusVaccine_TumorBinary"],
}

LABEL_MAPPING_FUNCTION = {
    "Solubility": ["Insoluble", "Soluble"],
    "DeepLocBinary": ["Membrane", "Soluble"],
    "DeepLocMulti": [
        "Cytoplasm", "Nucleus", "Extracellular", "Mitochondrion", "Cell membrane",
        "Endoplasmic reticulum", "Plastid", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome"
    ],
    "Metal ion binding": ["Non-binding", "Binding"],
    "Sortingsignal": ['No signal', "CH", 'GPI', "MT", "NES", "NLS", "PTS", "SP", "TM", "TH"],
    "SortingSignal": ['No signal', 'Signal'],
    "Varius": ["Non-virus", "Virus"],
    "Bacteria": ["Non-bacteria", "Bacteria"],
    "Tumor": ["Non-tumor", "Tumor"],
}

COLOR_MAP_FUNCTION = {
    "Soluble": "#3B82F6", "Insoluble": "#EF4444", "Membrane": "#F59E0B", 
    "Cytoplasm": "#10B981", "Nucleus": "#8B5CF6", "Extracellular": "#F97316", 
    "Mitochondrion": "#EC4899", "Cell membrane": "#6B7280", "Endoplasmic reticulum": "#84CC16", 
    "Plastid": "#06B6D4", "Golgi apparatus": "#A78BFA", "Lysosome/Vacuole": "#FBBF24", 
    "Peroxisome": "#34D399", "Binding": "#3B82F6", "Non-binding": "#EF4444", 
    "Bacteria": "#3B82F6", "Non-bacteria": "#EF4444", "Virus": "#3B82F6", "Non-virus": "#EF4444",
    "Tumor": "#3B82F6", "Non-tumor": "#EF4444", "Default": "#9CA3AF"
}

PROTEIN_PROPERTIES_FUNCTION = {
    "Physical and chemical properties",
    "Relative solvent accessible surface area (PDB only)",
    "SASA value (PDB only)",
    "Secondary structure (PDB only)"
}

PROTEIN_PROPERTIES_MAP_FUNCTION = {
    "Physical and chemical properties": "calculate_physchem",
    "Relative solvent accessible surface area (PDB only)": "calculate_rsa",
    "SASA value (PDB only)": "calculate_sasa",
    "Secondary structure (PDB only)": "calculate_secondary_structure"
}

RESIDUE_MAPPING_FUNCTION = {
    "Activity Site": ["VenusX_Res_Act_MP90"],
    "Binding Site": ["VenusX_Res_BindI_MP90"],
    "Conserved Site": ["VenusX_Res_Evo_MP90"],
    "Motif": ["VenusX_Res_Motif_MP90"]
}

REGRESSION_TASKS_FUNCTION = ["Stability", "Optimum temperature", "Kcat", "PH"]

REGRESSION_TASKS_FUNCTION_MAX_MIN = {
    "Stability": [40.1995166, 66.8968874],
    "Optimum temperature": [2, 120]
}

DATASET_TO_TASK_MAP = {
    dataset: task 
    for task, datasets in DATASET_MAPPING_FUNCTION.items() 
    for dataset in datasets
}

AI_MODELS = {
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1", 
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY"
    },
    "ChatGPT": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini", 
        "env_key": None
    },
    "Gemini": {
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-1.5-flash",
        "env_key": None
    }
}

