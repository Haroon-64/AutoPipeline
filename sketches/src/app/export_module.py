
class ExportModule:
    def __init__(self, config):
        self.config = config

    def export_script(self):
        script = self.generate_script()
        with open("generated_script.py", "w") as f:
            f.write(script)
        return "generated_script.py"

    def generate_script(self):
        task = self.config["task"]
        model_type = self.config["model_type"]
        data_path = self.config["data_path"]
        data_format = self.config["data_format"]
        num_classes = self.config["num_classes"]
        custom_layers = self.config["custom_layers"]
        epochs = self.config["epochs"]

        if model_type in ["decision_tree", "random_forest"]:
            return self.generate_sklearn_script(task, model_type, data_path, data_format)
        else:
            return self.generate_pytorch_script(task, model_type, data_path, data_format, num_classes, custom_layers, epochs)

    def generate_sklearn_script(self, task, model_type, data_path, data_format):
        model_class = "DecisionTreeClassifier" if model_type == "decision_tree" else "RandomForestClassifier"
        if task == "regression":
            model_class = model_class.replace("Classifier", "Regressor")

        script = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.{model_type} import {model_class}
import joblib

# Load data
df = pd.read_csv("{data_path}")
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train model
model = {model_class}()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.joblib")
"""
        return script

    def generate_pytorch_script(self, task, model_type, data_path, data_format, num_classes, custom_layers, epochs):
        script = f"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Load data
df = pd.read_csv("{data_path}")
data = df.drop('target', axis=1).values
labels = df['target'].values
train_data = CustomDataset(data, labels)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Define model
if "{model_type}" == "custom":
    layers = []
    for layer in {custom_layers}:
        if layer["type"] == "linear":
            layers.append(nn.Linear(layer["in_features"], layer["out_features"]))
        # Add more layer types as needed
    model = nn.Sequential(*layers)
else:
    # Add other model types here
    pass

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range({epochs}):
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), "model.pth")
"""
        return script