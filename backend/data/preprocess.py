from torchvision import transforms as vtransforms
from torchaudio import transforms as atransforms
from torchtext import transforms as ttransforms
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional


class TransformConfig(BaseModel):
  task: str
  transf: Optional[List[Dict[str, Dict]]] = None  # User-defined transforms

def get_default_transforms(config: TransformConfig):
  task = config.task.lower()
  if task == 'image':
    return {"transforms": [
      {"Resize": {"size": [256, 256]}},
      {"ToTensor": {}}
    ]}
  elif task == 'audio':
    return {"transforms": [
      {"Resample": {"orig_freq": 44100, "new_freq": 16000}},
      {"MelSpectrogram": {"sample_rate": 16000, "n_mels": 128}},
      {"FrequencyMasking": {"freq_mask_param": 30}},
      {"TimeMasking": {"time_mask_param": 100}}
    ]}
  elif task == 'text':
    return {"transforms": [
      {"VocabTransform": {"vocab": "your_vocab"}},
      {"Truncate": {"max_seq_len": 128}}
    ]}
  else:
    raise HTTPException(status_code=400, detail="Invalid task")

def build_transforms(config: TransformConfig):
  if config.transf is None:  # Use defaults if no user-defined transforms
    return get_default_transforms(config)
  
  # Return user-defined transforms directly as a JSON-compatible response
  return {"task": config.task.lower(), "transforms": config.transf}
