from collections import OrderedDict
import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision.models.vgg as torch_vgg
from torchvision.models.vgg import cfgs, make_layers
from torchvision.models._utils import _ovewrite_named_param
from utils.saver import SaveOutput


class VGG(torch_vgg.VGG):
    def __init__(self, features, **kwargs):
        super(VGG, self).__init__(features, **kwargs)

        self.save_output = SaveOutput()
        # regions we'll record from
        self.hook_map = OrderedDict([
            ("B1", ["features", 3]),
            ("B2", ["features", 8]),
            ("B3", ["features", 6]),
            ("B4", ["features", 17]),
            ("B5", ["features", 26]),
            ("TCL", ["features", 35]),
            ("POOL", ["avgpool"]),
        ])
        for name, address in self.hook_map.items():
            layer = self
            for entry in address:
                if isinstance(entry, str):
                    layer = getattr(layer, entry)
                elif isinstance(entry, int):
                    layer = layer[entry]
            layer.register_forward_hook(self.save_output.save_activation(name))

def _vgg(cfg: str, batch_norm: bool, weights, progress: bool, **kwargs) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model

def vgg19(*, weights = None, progress: bool = True, **kwargs) -> VGG:
    weights = torch_vgg.VGG19_Weights.verify(weights)
    return _vgg("E", False, weights, progress, **kwargs)
