from fastapi import FastAPI
from data.preprocess import build_transforms, TransformConfig
from data.data_loader import DataLoader
from models.pretrained import ModelLoader
from models.custom import CustomModel
from training.trainer import trainer
from tasks.libs import install_libraries

app = FastAPI()

@app.post("/install_libraries")
def install_libraries(task: str):
    return install_libraries(task)

@app.post("/train")
def train_model(config: dict):
    model = ModelLoader(config['task'], config['model'], config['version'], config['pretrained']).load_model()
    data_loader = DataLoader(config['data_dir'], config['batch_size'], config['image_size'], config['task']).get_data_loader(config['transform'])
    return trainer(model, data_loader, data_loader, config['criterion'], config['optimizer'], config['device'], config['epochs'])

@app.post("/load_model")
def load_model(config: dict):
    return ModelLoader(config['task'], config['model'], config['version'], config['pretrained']).load_model()

@app.post("/load_custom_model")
def load_custom_model(config: dict):
    return CustomModel(config['task'], config['model'], config['version']).load_model()

@app.post("/get_data_loader")
def get_data_loader(config: dict):
    return DataLoader(config['data_dir'], config['batch_size'], config['image_size'], config['task']).get_data_loader(config['transform'])



@app.post("/get_transforms")
def get_transforms(config: TransformConfig):
  return build_transforms(config)
