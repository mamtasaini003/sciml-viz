import torch
from collections import OrderedDict
import os

os.makedirs('models', exist_ok=True)

# Generate dummy FNO state dict
fno_state_dict = OrderedDict([
    ('spectral_conv_1.weights1', torch.randn(16, 16, 12, dtype=torch.cfloat)),
    ('spectral_conv_1.weights2', torch.randn(16, 16, 12, dtype=torch.cfloat)),
    ('mlp_1.weight', torch.randn(32, 16)),
    ('mlp_1.bias', torch.randn(32)),
    ('mlp_2.weight', torch.randn(1, 32)),
    ('mlp_2.bias', torch.randn(1)),
])

torch.save(fno_state_dict, 'models/sample_fno.pth')

# Generate dummy DeepONet state dict
deeponet_state_dict = OrderedDict([
    ('branch.layer1.weight', torch.randn(64, 100)),
    ('branch.layer1.bias', torch.randn(64)),
    ('trunk.layer1.weight', torch.randn(64, 2)),
    ('trunk.layer1.bias', torch.randn(64)),
    ('output.weight', torch.randn(1, 64)),
])

torch.save(deeponet_state_dict, 'models/sample_deeponet.pth')
