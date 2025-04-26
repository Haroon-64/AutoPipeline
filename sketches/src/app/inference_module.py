import torch
import torch.nn as nn

class InferenceModule:
    def __init__(self, model, model_path="model.pth"):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(model, nn.Module):
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)
            self.model.eval()

    def infer(self, data):
        if isinstance(self.model, nn.Module):
            with torch.no_grad():
                input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
                if len(input_tensor.shape) == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                output = self.model(input_tensor)
                return output.cpu().tolist()
        else:
            return self.model.predict([data])