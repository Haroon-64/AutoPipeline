import torch
from torchviz import make_dot
import graphviz
from sklearn.tree import export_graphviz

class VisualizeModule:
    def __init__(self, model):
        self.model = model

    def visualize(self):
        if isinstance(self.model, torch.nn.Module):  # PyTorch model
            sample_input = torch.randn(1, 3, 224, 224)  # Example input size for image models
            output = self.model(sample_input)
            dot = make_dot(output, params=dict(self.model.named_parameters()))
            dot.render("model_graph", format="png")
            return "model_graph.png"
        elif hasattr(self.model, 'tree_'):  # Scikit-learn decision tree
            dot_data = export_graphviz(self.model, out_file=None, filled=True)
            graph = graphviz.Source(dot_data)
            graph.render("tree_graph", format="png")
            return "tree_graph.png"
        else:
            return None  # Unsupported model type