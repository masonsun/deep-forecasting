import torch

opts = {
    # file names
    'data_path': './data',
    'vae_state': 'vae_ckpt.pth',
    'nn_state': 'nn_ckpt.pth',

    # configuration
    'train_per': 0.85,     # training split
    'dropout': 0.5,        # probability of dropout
    'lr': 1e-3,            # learning rate
    'lrd': 1e-6,           # learning rate decay
    'w_decay': 5e-4,       # weight decay
    'momentum': 0.9,       # momentum
    'grad_clip': 10,       # clip gradient
    'beta': (0.95, 0.95),  # Adam beta parameters
    'fudge': 1e-7,         # fudge factor for numerical stability

    # gpu
    'use_cuda': True and torch.cuda.is_available()
}
