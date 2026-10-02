import torch
from torch import nn
from torch.nn import functional as F
import importlib

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def update_config(args, config):
    """Apply CLI values used by configuration interpolation."""
    from pathlib import Path
    for key, value in vars(args).items():
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    return config


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='none')


        return image_loss,text_loss, logits_per_image

def load_blur_images(blur_paths, device='cpu'):
    """Stack paired image tensors or unpack two batched tensors, then move them to device."""
    if isinstance(blur_paths, list):
        if len(blur_paths) == 0:
            raise ValueError("blur_paths list is empty!")

        if isinstance(blur_paths[0], tuple):
            blur_a_list = [item[0] for item in blur_paths]
            blur_b_list = [item[1] for item in blur_paths]
            blur_a = torch.stack(blur_a_list)
            blur_b = torch.stack(blur_b_list)
        elif len(blur_paths) == 2 and isinstance(blur_paths[0], torch.Tensor):
            blur_a, blur_b = blur_paths[0], blur_paths[1]
        else:
            raise TypeError(f"list Invalid format: {type(blur_paths[0])}")

    elif isinstance(blur_paths, tuple):
        if len(blur_paths) != 2:
            raise ValueError(f"tuple Invalid length: {len(blur_paths)}")
        blur_a, blur_b = blur_paths

    else:
        raise TypeError(f"Unsupported input type: {type(blur_paths)}")

    if not isinstance(blur_a, torch.Tensor) or not isinstance(blur_b, torch.Tensor):
        raise TypeError(f"Expected tensor outputs: blur_a={type(blur_a)}, blur_b={type(blur_b)}")

    return blur_a.to(device), blur_b.to(device)
