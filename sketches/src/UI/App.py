import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import gradio as gr
from src.app.data_module import DataModule
from src.app.model_module import ModelModule
from src.app.train_module import TrainModule
from src.app.inference_module import InferenceModule
from src.app.visualize_module import VisualizeModule
from src.app.export_module import ExportModule
import json

def train_model(task, data_path, data_format, model_type, num_classes, custom_layers_json, epochs):
    config = {
        "task": task,
        "data_path": data_path,
        "data_format": data_format,
        "model_type": model_type,
        "num_classes": num_classes,
        "custom_layers": json.loads(custom_layers_json),
        "epochs": epochs
    }
    data_module = DataModule(task, data_path, data_format)
    model_module = ModelModule(model_type, task, num_classes, config["custom_layers"])
    model = model_module.get_model()
    train_loader, test_loader = data_module.get_loaders()
    train_module = TrainModule(model, train_loader, test_loader, task, epochs=epochs)
    result = train_module.train()
    return result, model, config

def infer_data(model, input_data):
    inference_module = InferenceModule(model)
    result = inference_module.infer(eval(input_data))
    return str(result)

def visualize_model(model):
    visualize_module = VisualizeModule(model)
    img_path = visualize_module.visualize()
    return img_path if img_path else gr.update(visible=False)

def export_script(config):
    export_module = ExportModule(config)
    script_path = export_module.export_script()
    return script_path

with gr.Blocks() as app:
    gr.Markdown("# ML Pipeline Designer")

    # Inputs
    task = gr.Dropdown(["classification", "regression", "image_classification", "text_classification"], label="Task")
    data_path = gr.Textbox(label="Data Path")
    data_format = gr.Dropdown(["csv", "image_folder", "text"], label="Data Format")
    model_type = gr.Radio(["decision_tree", "random_forest", "cnn", "nlp", "custom"], label="Model Type")
    num_classes = gr.Number(label="Number of Classes", value=2)
    custom_layers = gr.Textbox(label="Custom Layers JSON", value="[]")
    epochs = gr.Slider(1, 50, value=10, label="Epochs")

    # Buttons and Outputs
    train_btn = gr.Button("Train Model")
    output_text = gr.Textbox(label="Training Output")
    trained_model = gr.State()
    config_state = gr.State()

    infer_input = gr.Textbox(label="Input Data for Inference")
    infer_btn = gr.Button("Run Inference")
    infer_output = gr.Textbox(label="Inference Output")

    viz_btn = gr.Button("Visualize Model")
    viz_output = gr.Image(label="Model Visualization")

    export_btn = gr.Button("Export Script")
    export_output = gr.File(label="Download Script")

    # Event Handlers
    train_btn.click(
        fn=train_model,
        inputs=[task, data_path, data_format, model_type, num_classes, custom_layers, epochs],
        outputs=[output_text, trained_model, config_state]
    )
    infer_btn.click(
        fn=infer_data,
        inputs=[trained_model, infer_input],
        outputs=infer_output
    )
    viz_btn.click(
        fn=visualize_model,
        inputs=trained_model,
        outputs=viz_output
    )
    export_btn.click(
        fn=export_script,
        inputs=config_state,
        outputs=export_output
    )

app.launch()