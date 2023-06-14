import torch


def load_model(model_name):
    return torch.hub.load("ultralytics/yolov5", "custom", path=f"{model_name}.pt")
