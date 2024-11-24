import gradio as gr

def create_eval_tab(constant):
    dataset_configs = constant["dataset_configs"]

    def evaluate_model(model_path, dataset, batch_size):
        # TODO: Implement evaluation logic
        return f"Evaluating model {model_path} on dataset {dataset} with batch size {batch_size}"

    with gr.Tab("Evaluation"):
        gr.Markdown("## Model Evaluation")
        with gr.Row():
            with gr.Column():
                eval_model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="Path to the trained model"
                )
                eval_dataset = gr.Dropdown(
                    choices=list(dataset_configs.keys()),
                    label="Evaluation Dataset"
                )
            with gr.Column():
                eval_batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=32,
                    step=1,
                    label="Evaluation Batch Size"
                )
        
        eval_button = gr.Button("Start Evaluation")
        eval_output = gr.Textbox(label="Evaluation Results", lines=10)

        # Bind evaluation event
        eval_button.click(
            fn=evaluate_model,
            inputs=[eval_model_path, eval_dataset, eval_batch_size],
            outputs=eval_output
        )

        return {
            "eval_button": eval_button,
            "eval_output": eval_output
        }