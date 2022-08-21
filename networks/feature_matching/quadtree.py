import sys
sys.path.append('./networks/feature_matching')
import torch
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from src.loftr import LoFTR

def load_network(device, path=None, main_config=None):
    config = get_cfg_defaults()
    config.merge_from_file(main_config)
    _config = lower_config(config)
    model = LoFTR(config=_config['loftr'])
    state_dict = torch.load(path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device)
    return model