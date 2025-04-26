import torch
import torch.nn as nn

class CustomLoader(nn.module):
    def __init__(self) -> None:
        super(CustomLoader,self).__init__()
        self.layers = nn.ModuleList() 
    # TODO: implement custom model loader to create layers


