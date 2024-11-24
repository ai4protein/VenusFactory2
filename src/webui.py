import json
import time
import gradio as gr
from web.monitor import TrainingMonitor
from web.train_tab import create_train_tab
from web.eval_tab import create_eval_tab

def create_ui():
    # Load configuration
    constant = json.load(open("src/constant.json"))
    monitor = TrainingMonitor()

    def update_output():
        if monitor.is_training:
            messages = monitor.get_messages()
            progress = monitor.get_progress()
            return messages, gr.update(value=progress, visible=True)
        return "", gr.update(value="", visible=False)

    with gr.Blocks() as demo:
        gr.Markdown("# ProFactory Training Interface")
        
        with gr.Tabs():
            # Create training tab
            train_components = create_train_tab(monitor, constant)
            
            # Create evaluation tab
            eval_components = create_eval_tab(constant)

            # Bind training button event
            def train_with_batch_mode(
                model_name,
                dataset_config,
                training_method,
                batch_mode,
                size_value,
                token_value,
                *args
            ):
                mode = "size" if batch_mode == "Batch Size Mode" else "token"
                value = size_value if mode == "size" else token_value
                
                return train_components["train_fn"](
                    model_name,
                    dataset_config,
                    training_method,
                    mode,
                    value,
                    *args
                )

            train_components["train_button"].click(
                fn=train_with_batch_mode,
                inputs=[
                    *train_components["components"].values()
                ],
                outputs=[train_components["output_text"]]
            )


            demo.load(
                fn=update_output,
                inputs=None,
                outputs=[
                    train_components["output_text"],
                    train_components["progress_bar"]
                ]
            )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", share=True)