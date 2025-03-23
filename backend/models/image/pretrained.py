import torch
import torchvision.models as vmodels

# from torchvision.models import (
#     resnet18, resnet34, resnet50, resnet101, resnet152,
#     mobilenet_v3_small, mobilenet_v3_large,
#     ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
#     ResNet101_Weights, ResNet152_Weights, MobileNet_V2_Weights, MobileNet_V3_Large_Weights
# )

model_configs = {
    "resnet": {"50": vmodels.resnet50, "101": vmodels.resnet101},
    "efficientnet": {"b0": vmodels.efficientnet_b0, "b1": vmodels.efficientnet_b1},
    "inceptionnet": {"default": vmodels.inception_v3},
    "vgg": {"default": vmodels.vgg16},
    "densenet": {"default": vmodels.densenet121},
    "alexnet": {"default": vmodels.alexnet},
    "squeezenet": {"default": vmodels.squeezenet1_0},
    "mobilenet": {"default": vmodels.mobilenet_v2},
    "googlenet": {"default": vmodels.googlenet},
    "shufflenet": {"default": vmodels.shufflenet_v2_x1_0},
    "mnasnet": {"default": vmodels.mnasnet1_0},

}


class ModelLoader:
    """
    ModelLoader class to load pretrained models
    models:
    - resnet
    - efficientnet
    - inceptionnet
    - vgg
    - densenet
    - alexnet
    - squeezenet
    - mobilenet
    - googlenet
    - shufflenet
    - mnasnet    
    """
    def __init__(self, task, name, version=None, pretrained=True):
        self.task = task
        self.name = name.lower()
        self.version = version
        self.pretrained = pretrained

    def load_model(self):
        model = None

        if self.name in model_configs:
            model_fn = model_configs[self.name].get(self.version, None)
            if model_fn:
                model = model_fn(pretrained=self.pretrained)

        if model is not None and torch.cuda.is_available():
            model = model.to('cuda')

        return model
    


"""
loader = ModelLoader(task='image', name='resnet', version="50", pretrained=True)
model = loader.load_model()
"""


def _gettfmodel(self, name: str, size, pretrained):
    # if tf models are needed  
    model = None
    pass
