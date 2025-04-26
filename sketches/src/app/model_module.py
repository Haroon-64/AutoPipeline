import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torchvision import models
from transformers import AutoModelForSequenceClassification

class ModelModule:
    def __init__(self, model_type, task, num_classes=None, custom_layers=None, optimizer="adam", activation="relu"):
        self.model_type = model_type
        self.task = task
        self.num_classes = num_classes if num_classes else 2
        self.custom_layers = custom_layers or []
        self.optimizer = optimizer
        self.activation = activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()

    def build_model(self):
        if self.model_type == "decision_tree":
            if self.task == "classification":
                return DecisionTreeClassifier()
            elif self.task == "regression":
                return DecisionTreeRegressor()
        elif self.model_type == "random_forest":
            if self.task == "classification":
                return RandomForestClassifier()
            elif self.task == "regression":
                return RandomForestRegressor()
        elif self.model_type == "cnn":
            if self.task == "image_classification":
                model = models.resnet18(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                return model.to(self.device)
            else:
                raise ValueError(f"CNN not supported for task: {self.task}")
        elif self.model_type == "nlp":
            if self.task == "text_classification":
                model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.num_classes)
                return model.to(self.device)
            else:
                raise ValueError(f"NLP not supported for task: {self.task}")
        elif self.model_type == "custom":
            if not self.custom_layers:
                raise ValueError("Custom layers must be provided")
            layers = []
            for layer in self.custom_layers:
                if layer["type"] == "linear":
                    layers.append(nn.Linear(layer["in_features"], layer["out_features"]))
                elif layer["type"] == "conv2d":
                    layers.append(nn.Conv2d(layer["in_channels"], layer["out_channels"], layer["kernel_size"]))
                # Add activation after each layer
                if self.activation == "relu":
                    layers.append(nn.ReLU())
                elif self.activation == "sigmoid":
                    layers.append(nn.Sigmoid())
            # Output layer based on task
            if self.task in ["classification", "image_classification", "text_classification"]:
                layers.append(nn.Linear(layers[-2].out_features, self.num_classes))
            elif self.task == "regression":
                layers.append(nn.Linear(layers[-2].out_features, 1))
            model = nn.Sequential(*layers)
            return model.to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def get_model(self):
        return self.model