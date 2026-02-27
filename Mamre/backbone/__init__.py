BACKBONES = {}

try:
    from ._mamba_ssm import MambaSSMMamreBackbone

    BACKBONES["mamba_ssm"] = MambaSSMMamreBackbone
except ImportError:
    pass

from ._torch import TorchMamreBackbone

BACKBONES["torch"] = TorchMamreBackbone
