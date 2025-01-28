import torch
import torchaudio.models as amodels
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
    "demucs": {
        "default": amodels.HDemucs,
        "tasnet": amodels.TasNet,
        "conv_tasnet": amodels.ConvTasNet,
        "dprnn_tasnet": amodels.DPRNNTasNet,
    },
    "unet": {"default": amodels.UNet},
    "transformer": {"default": amodels.Transformer},
    "transformerlm": {"default": amodels.TransformerLM},
    "wav2vec": {"default": amodels.Wav2Vec},

    "wavernn": {"default": amodels.WaveRNN},
    "conformer": {"default": amodels.Conformer},
    "wav2vec2": {"default": amodels.Wav2Vec2},
    "wav2letter": {"default": amodels.Wav2Letter},
    "gpt2": {"small": amodels.GPT2, "medium": amodels.GPT2, "large": amodels.GPT2, "xlarge": amodels.GPT2},
    "bert": {"base": amodels.BERT, "large": amodels.BERT},
    "roberta": {"base": amodels.RoBERTa, "large": amodels.RoBERTa},
    "distilbert": {"base": amodels.DistilBERT, "large": amodels.DistilBERT},
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
    - demucs
    - unet
    - transformer
    - transformerlm
    - wav2vec
    - wavernn
    - conformer
    - wav2vec2
    
    """
    def __init__(self, task, name, version=None, pretrained=True):
        self.task = task
        self.name = name.lower()
        self.version = version
        self.pretrained = pretrained

    def load_model(self):
        model = None

        if self.task == "image":
            if self.name in model_configs and self.name in vmodels.__dict__:
                model_fn = model_configs[self.name].get(
                    self.version, None
                ) or model_configs[self.name].get("default", None)
                if model_fn:
                    model = model_fn(pretrained=self.pretrained)

        elif self.task == "audio":
            if self.name in model_configs and self.name in amodels.__dict__:
                model_fn = model_configs[self.name].get("default", None)
                if model_fn:
                    model = model_fn(pretrained=self.pretrained)

        elif self.task == "text":
            if self.name in model_configs and self.name in amodels.__dict__:
                model_fn = model_configs[self.name].get(
                    self.version, None
                ) or model_configs[self.name].get("default", None)
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
