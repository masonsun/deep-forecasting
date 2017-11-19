import torch

opts = {
    # configuration
    'use_cuda': True and torch.cuda.is_available(),

    # hyper-parameters
    'dropout': 0.5,
    'frame': 7,

    # numerical assignments
    'fudge': 1e-7
}
