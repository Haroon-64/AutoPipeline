import torch
import torchaudio.models as amodels


model_configs = {
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
    ModelLoader class to load pretrained audio models
    models:
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
        if self.name in model_configs and self.name in amodels.__dict__:
            model_fn = model_configs[self.name].get("default", None)
            if model_fn:
                model = model_fn(pretrained=self.pretrained)   #  TODO: adjust for different models and versions

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
