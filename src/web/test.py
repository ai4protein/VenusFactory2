import gradio as gr
from gradio_molecule3d import Molecule3D

reps = [
    {
        "model": 0,
        "chain": "",
        "resname": "",
        "style": "cartoon",
        "color": "pLDDT",
        "byres": True,
        "around": 0,
    }
]

def display_molecule(molecule_file):
    return molecule_file

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.HTML(create_plddt_legend())

    with gr.Row():
        inp = Molecule3D(
            label="Input Molecule (PDB/CIF)",
            reps=reps
        )
    btn = gr.Button("Render with pLDDT Color", variant="primary")
    btn.click(display_molecule, inputs=inp)
if __name__ == "__main__":
    demo.launch()
