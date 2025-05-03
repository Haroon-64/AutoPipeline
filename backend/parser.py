import json
from jinja2 import Environment, FileSystemLoader

# Load pipeline definition
with open("pipeline.txt") as f:
    pipeline = json.load(f)

# Unpack values
imports = [pipeline[0]]
task = {
    "ml_or_dl": pipeline[1],
    "task_type": pipeline[2],
    "data_type": pipeline[3],
    "data_path": '',
}

model = {
    "framework": "pytorch" if pipeline[1] == "dl" else "sklearn",
    "prebuilt_model": pipeline[5] if pipeline[4] == "pretrained" else None,
    "use_pretrained": True if pipeline[4] == "pretrained" else False,
    "custom_layers": pipeline[5] if pipeline[4] == "custom" else []
}

training = {
    "loss_function": pipeline[6],
    "optimizer": pipeline[7],
    "learning_rate": pipeline[8],
    "epochs": pipeline[9]
}

env = Environment(loader=FileSystemLoader('.'))

# Render code template
code_template = env.get_template("template.j2")
rendered_code = code_template.render(
    imports=imports,
    task=task,
    model=model,
    training=training
)

with open("output.py", "w") as f:
    f.write(rendered_code)

# Render documentation template
doc_template = env.get_template("doc_template.j2")
rendered_doc = doc_template.render(
    task=task,
    model=model,
    training=training
)

with open("documentation.md", "w") as f:
    f.write(rendered_doc)
