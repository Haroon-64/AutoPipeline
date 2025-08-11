import random
from pathlib import Path

import numpy
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        label_type="folder-name",
        label_map=None,
        return_format="dict",
        split_type="include",
    ):
        self.root = Path(root) / split
        self.label_map = self._infer_label_map()
        self.return_format = return_format
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.data = self._load_data()

    def _infer_label_map(self):
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        return {cls: i for i, cls in enumerate(classes)}

    def _load_data(self):
        items = []
        for cls in self.label_map:
            for img_path in (self.root / cls).glob("*.jpg"):
                items.append((img_path, self.label_map[cls]))
        return items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))

        return {"image": image, "label": label}


def get_loader(root, batch_size=32, split="train", shuffle=True, **kwargs):
    dataset = ImageClassificationDataset(root=root, split=split, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


import random

import numpy as np
import torch


def set_seed(seed=None):
    seed = seed or config.training.get("seed", {{config.get("SEED", 42)}})
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device=None):
    device_str = device or config.get("DEVICE", "cpu")
    return torch.device(device_str if torch.cuda.is_available() else "cpu")


import torch.nn as nn
import torch.optim as optim


def get_criterion(config):
    loss_cfg = config.loss
    name = loss_cfg.name
    params = loss_cfg.get("params", {})
    return getattr(nn, name)(**params)


def get_optimizer(model, config):
    opt = config.optimizer
    Optim = getattr(optim, opt.name)
    params = opt.get("params", {})
    return Optim(model.parameters(), lr=config.training.learningRate, **params)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for batch in loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


# Optimizer definition


optimizer = torch.optim.Adam(model.parameters(), **{})


# Loss function definition


criterion = torch.nn.CrossEntropyLoss(**{})


metrics = []


metrics.append(torchmetrics.Accuracy(**{}))
