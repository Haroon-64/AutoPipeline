import torchvision.models as vmodels
import torchaudio.models as amodels  # Placeholder for the actual audio models

model_configs = {
  "resnet": {
    "50": vmodels.resnet50,
    "101": vmodels.resnet101
  },
  "efficientnet": {
    "b0": vmodels.efficientnet_b0,
    "b1": vmodels.efficientnet_b1
  },
  "inceptionnet": {
    "default": vmodels.inception_v3
  },
  "demucs": {
    "default": amodels.HDemucs  # Example audio models
  },
  "wavernn": {
    "default": amodels.WaveRNN
  },
  "conformer": {
    "default": amodels.Conformer
  }
}

class ModelLoader:
  def __init__(self, task, name, version=None, pretrained=True):
    self.task = task
    self.name = name.lower()
    self.version = version
    self.pretrained = pretrained

  def load_model(self):
    model = None
    
    if self.task == 'image':
      if self.name in model_configs:
        model_fn = model_configs[self.name].get(self.version, None) or model_configs[self.name].get("default", None)
        if model_fn:
          model = model_fn(pretrained=self.pretrained)
    
    elif self.task == 'audio':
      if self.name in model_configs:
        model_fn = model_configs[self.name].get("default", None)
        if model_fn:
          model = model_fn()  # Adjust based on audio models' needs for pre-trained weights

    return model


"""
loader = ModelLoader(task='image', name='resnet', version="50", pretrained=True)
model = loader.load_model()
"""

def _gettfmodel(self,name:str,size,pretrained): 
    model = None
    pass
