import torch

opts = {
    # file paths
    'data_path': './data',

    # file names
    'vae_state': 'vae_ckpt.pth',
    'nn_state': 'nn_ckpt.pth',

    # configuration
    'plot': 99,           # at which epoch to plot graphs

    # splits
    'train_per': 0.90,    # training split
    'valid_per': 0.05,    # validation split (rest is testing)

    # training
    'test_frequency': 5,  # how often to evaluate test set
    'dropout': 0.5,       # probability of dropout

    # tuning
    'lr': 1e-3,            # learning rate
    'lrd': 1e-6,           # learning rate decay
    'w_decay': 5e-4,       # weight decay
    'momentum': 0.9,       # momentum
    'grad_clip': 10,       # clip gradient
    'beta': (0.95, 0.95),  # Adam beta parameters

    # other
    'fudge': 1e-7,         # fudge factor for numerical stability

    # gpu
    'use_cuda': True and torch.cuda.is_available()
}
