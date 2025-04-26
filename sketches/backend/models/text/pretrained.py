import torch
import torchtext.models as tmodels

# from torchvision.models import (
#     resnet18, resnet34, resnet50, resnet101, resnet152,
#     mobilenet_v3_small, mobilenet_v3_large,
#     ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
#     ResNet101_Weights, ResNet152_Weights, MobileNet_V2_Weights, MobileNet_V3_Large_Weights
# )

model_configs = {
    "transformer": {"default": tmodels.Transformer},
    "transformerlm": {"default": tmodels.TransformerLM},
    "gpt2": {"small": tmodels.GPT2, "medium": tmodels.GPT2, "large": tmodels.GPT2, "xlarge": tmodels.GPT2},
    "bert": {"base": tmodels.BERT, "large": tmodels.BERT},
    "roberta": {"base": tmodels.RoBERTa, "large": tmodels.RoBERTa},
    "distilbert": {"base": tmodels.DistilBERT, "large": tmodels.DistilBERT},
    "albert": {"base": tmodels.ALBERT, "large": tmodels.ALBERT},
    "rnn": {"default": tmodels.RNN},
    "lstm": {"default": tmodels.LSTM},
    "gru": {"default": tmodels.GRU},
    "cnn": {"default": tmodels.CNN},
}


class ModelLoader:
    """
    ModelLoader class to load pretrained models
    models:
    - transformer
    - transformerlm
    - gpt2
    - bert
    - roberta
    - distilbert
    - albert
    - rnn
    - lstm
    - gru
    - cnn
    
    """
    def __init__(self, task, name, version=None, pretrained=True):
        self.task = task
        self.name = name.lower()
        self.version = version
        self.pretrained = pretrained

    def load_model(self):
        model = None
        if self.name in model_configs:
            model_fn = model_configs[self.name].get("default", None)
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
