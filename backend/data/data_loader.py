import os
import torch
from torchvision import dataset as vdatasets
from torchaudio import datasets as adatasets
from torchtext import data as tdata

class DataLoader:
    def __init__(self, data_dir, batch_size, image_size,task):
        
        self.task = task
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

    def get_data_loader(self,transform):

        if self.task == 'image':
            dataset = vdatasets.ImageFolder(root=self.data_dir, transform=transform)
        elif self.task == 'audio':
            dataset = adatasets.SPEECHCOMMANDS(root=self.data_dir, download=True)
        elif self.task == 'text':# dummy - check to specications 
            TEXT = tdata.Field()   
            LABEL = tdata.Field(sequential=False)
            dataset = tdata.TabularDataset(
                path=self.data_dir, format='csv',
                fields=[('text', TEXT), ('label', LABEL)],
                skip_header=True
            )
            TEXT.build_vocab(dataset)
            LABEL.build_vocab(dataset)
        else:
            raise ValueError("Invalid task")
        
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return data_loader
