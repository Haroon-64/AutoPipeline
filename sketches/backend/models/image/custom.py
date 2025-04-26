import torch
import torch.nn as nn

class CustomLoader(nn.module):
    def __init__(self) -> None:
        super(CustomLoader,self).__init__()
        self.layers = nn.ModuleList() 


    
    def add_layer(self,layer_type,*a,**k):
        layer = self._create_layer(layer_type,*a,**k)
        self.layers.append(layer)

    def _create_layer(self,layer_type:str,*a,**k):
        if layer_type.lower() == 'conv2d':
            return nn.Conv2d(*a,**k)
        if layer_type.lower() == 'linear':
            return nn.Linear(*a, **k)
        if layer_type.lower() == 'relu':
            return nn.ReLU(*a, **k)
        # TODO: add more  eg other non linearity, skip conns etc


    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def summary(self):
        print(self)
    


