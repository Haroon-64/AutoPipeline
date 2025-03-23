from torchaudio import transforms as atransforms
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional


class TransformConfig(BaseModel):
  task: str
  transf: Optional[List[Dict[str, Dict]]] = None 
  size: Optional[int] = 224
  sample_rate: Optional[List[int]] = [44100,16000] # [original, new]
  vocab: Optional[str] = "tmp"
  

def get_default_transforms(config: TransformConfig):
  try:
    return {"transforms": [
      {"Resample": {"orig_freq": config.sample_rate[0], "new_freq": config.sample_rate[1]}},
      {"MelSpectrogram": {"sample_rate": 16000, "n_mels": 128}},
      {"FrequencyMasking": {"freq_mask_param": 30}},
      {"TimeMasking": {"time_mask_param": 100}}
    ]}

  except(Exception):
    raise Exception("Invalid task")

def build_transforms(config: TransformConfig):
  if config.transf is None:  
    return get_default_transforms(config)
  
  # Return user-defined transforms directly as a JSON-compatible response
  return {"task": config.task.lower(), "transforms": config.transf}
