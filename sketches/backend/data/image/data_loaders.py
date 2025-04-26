import os
from torchvision import transforms as Tv
from torchvision.datasets import ImageFolder as Iv
from torch.utils.data import DataLoader as Dv, random_split

from torchaudio import transforms as Ta


def load_image_data(data_path: str,
              image_size: int = 224,
              test_size: float = 0.2,
              transform: str = None,
              BATCHSIZE: int = 32,
              val_size: float = 0.1,  
              include_val: bool = False):# -> dict[str, str] | dict[str, DataLoader]:  
    
    if not os.path.exists(data_path):
        return {"error": "Invalid data path"}
    
    # Setup transformations
    transform_list = [Tv.Resize((image_size, image_size))]
    if transform == "augmentation":
        transform_list.extend([Tv.RandomHorizontalFlip(), Tv.RandomRotation(15)])
    transform_list.append(Tv.ToTensor())
    
    try:
        # Load the full dataset
        full_dataset = Iv(root=data_path, transform=Ta.Compose(transform_list))
        num_classes = len(full_dataset.classes)

        # Split into train and test datasets
        train_size = int((1 - test_size) * len(full_dataset))  
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Further split train dataset into train and validation datasets if include_val is True
        if include_val:
            val_size = int(val_size * train_size)
            train_size = train_size - val_size  # Adjust size of training dataset after split
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            val_loader = Dv(val_dataset, batch_size=BATCHSIZE, shuffle=False)
            return {
                'train_loader': Dv(train_dataset, batch_size=BATCHSIZE, shuffle=True),
                'test_loader': Dv(test_dataset, batch_size=BATCHSIZE, shuffle=False),
                'val_loader': val_loader,
                'num_classes': num_classes
            }

        # If no validation is needed, return only train and test loaders
        return {
            'train_loader': Dv(train_dataset, batch_size=BATCHSIZE, shuffle=True),
            'test_loader': Dv(test_dataset, batch_size=BATCHSIZE, shuffle=False),
            'num_classes': num_classes
        }

    except Exception as e:
        return {"error": str(e)}    