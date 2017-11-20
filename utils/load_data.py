import sys
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.utils.data as utils

from collections import OrderedDict
from src.options import opts

# WARNING:
# Maybe we should consider contiguous time series only (and not simply remove NaNs)

# INFO:
# We should increase batch size if we can definitely say all games have enough data to allow it


def load_data(fp=os.path.join(opts['data_path'], opts['data_filename'])):
    """
    :param fp: relative path to data
    :return: OrderedDict of game_id --> (date, users)
    """
    try:
        data = pd.read_csv(fp)
    except IOError:
        print("File not found: {}".format(fp))
        sys.exit(-1)

    dates = data['date']
    data.drop('date', axis=1, inplace=True)

    data_dict = OrderedDict()
    min_length = np.inf
    for col in data:
        x = list(zip(dates, data[col]))
        x = [i for i in x if not any(isinstance(j, float) and math.isnan(j) for j in i)]
        min_length = len(x) if len(x) < min_length else min_length
        data_dict[col] = x

    return data_dict


def set_dataloader(x, y):
    """
    :param x: list of data
    :param y: list of targets
    :return: torch dataloader
    """
    x = torch.stack([torch.Tensor(i) for i in x])
    y = torch.stack([torch.Tensor(i) for i in y])
    dataset = utils.TensorDataset(x, y)

    kwargs = {'num_workers': 1, 'pin_memory': True} if opts['gpu'] else {}
    return utils.DataLoader(dataset, batch_size=opts['batch_size'], shuffle=False, **kwargs)
