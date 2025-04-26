import torch
import torch.optim as optim
import torch.nn as nn

class TrainModule:
    def __init__(self, model, train_loader, test_loader, task, epochs=10, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.task = task
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(model, nn.Module):
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr) if hasattr(model, "parameters") else None
            self.criterion = self.get_criterion()

    def get_criterion(self):
        if self.task in ["classification", "image_classification", "text_classification"]:
            return nn.CrossEntropyLoss()
        elif self.task == "regression":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def train(self):
        if isinstance(self.model, nn.Module):
            self.model.train()
            for epoch in range(self.epochs):
                for batch in self.train_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")
            torch.save(self.model.state_dict(), "model.pth")
        else:
            # For scikit-learn models
            X_train, y_train = self.train_loader.dataset.data, self.train_loader.dataset.labels
            self.model.fit(X_train, y_train)
            print("Model trained successfully.")
        return "Training complete."