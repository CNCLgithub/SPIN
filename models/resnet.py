from collections import OrderedDict
import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models._utils import _ovewrite_named_param
from utils.saver import SaveOutput


class ResNet(torch_resnet.ResNet):
    def __init__(self, block, layers, use_last_fc, **kwargs):
        super(ResNet, self).__init__(block, layers, **kwargs)
        self.use_last_fc = use_last_fc

        self.save_output = SaveOutput()
        # regions we'll record from
        self.hook_map = OrderedDict([
            ("B1U1", ["layer1", 0]),
            ("B1U2", ["layer1", 1]),
            ("B1U3", ["layer1", 2]),
            ("B2U1", ["layer2", 0]),
            ("B2U2", ["layer2", 1]),
            ("B2U3", ["layer2", 2]),
            ("B2U4", ["layer2", 3]),
            ("B3U1", ["layer3", 0]),
            ("B3U2", ["layer3", 1]),
            ("B3U3", ["layer3", 2]),
            ("B3U4", ["layer3", 3]),
            ("B3U5", ["layer3", 4]),
            ("B3U6", ["layer3", 5]),
            ("B4U1", ["layer4", 0]),
            ("B4U2", ["layer4", 1]),
            ("TCL", ["layer4", 2]),
            ("POOL", ["avgpool"]),
        ])
        for region, index in self.hook_map.items():
            layer = getattr(self, index[0])
            if len(index) == 2:
                layer = layer[index[1]]
            layer.register_forward_hook(self.save_output.save_activation(region))

def _resnet(block, layers, weights, progress, use_last_fc, **kwargs):
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, use_last_fc, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def resnet50(*, weights, progress = True, use_last_fc=True, **kwargs):
    weights = torch_resnet.ResNet50_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, use_last_fc, **kwargs)
