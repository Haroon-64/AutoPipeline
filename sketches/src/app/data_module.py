import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Custom dataset class to handle various data types
class CustomDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

class DataModule:
    def __init__(self, task, data_path, data_format, test_size=0.2):
        self.task = task
        self.data_path = data_path
        self.data_format = data_format
        self.test_size = test_size
        self.train_data, self.test_data = self.load_data()

    def load_data(self):
        if self.data_format == "csv":
            df = pd.read_csv(self.data_path)
            if self.task in ["classification", "regression"]:
                train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=42)
                train_data = CustomDataset(train_df.drop('target', axis=1).values, train_df['target'].values)
                test_data = CustomDataset(test_df.drop('target', axis=1).values, test_df['target'].values)
            elif self.task == "text_classification":
                train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=42)
                train_data = CustomDataset(train_df['text'].tolist(), train_df['label'].tolist())
                test_data = CustomDataset(test_df['text'].tolist(), test_df['label'].tolist())
            else:
                raise ValueError(f"Unsupported task for CSV: {self.task}")
        elif self.data_format == "image_folder":
            if self.task == "image_classification":
                transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
                dataset = datasets.ImageFolder(self.data_path, transform=transform)
                train_data, test_data = train_test_split(dataset, test_size=self.test_size, random_state=42)
            else:
                raise ValueError(f"Unsupported task for image folder: {self.task}")
        elif self.data_format == "text":
            # Placeholder for raw text (extendable for NLP tasks)
            raise NotImplementedError("Text format not yet implemented")
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")
        return train_data, test_data

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader