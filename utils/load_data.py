import sys
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.utils.data as utils

from collections import OrderedDict
from src.options import opts


def load_data(fp, verbose=False):
    """
    :param fp: relative path to data
    :param verbose: verbose mode
    :return: OrderedDict of game_id --> (date, users)
    """
    if not opts['data_path'].split('/')[-1] in fp:
        fp = os.path.join(opts['data_path'], fp)

    try:
        data = pd.read_csv(fp)
    except IOError:
        print("File not found: {}".format(fp))
        sys.exit(-1)

    dates = data['date']
    data.drop('date', axis=1, inplace=True)

    # what our data looks like
    if verbose:
        print("Dataset:\n{}\n".format(data.head(3)))

    data_dict = OrderedDict()
    min_length = np.inf
    for col in data:
        x = list(zip(dates, data[col]))
        x = [i for i in x if not any(isinstance(j, float) and math.isnan(j) for j in i)]
        min_length = len(x) if len(x) < min_length else min_length
        data_dict[col] = x

    return data_dict


def set_dataloader(x, y, batch_size=1):
    """
    :param x: list of data
    :param y: list of targets
    :param batch_size: batch size
    :return: torch dataloader
    """
    try:
        x = torch.stack([torch.Tensor(i.tolist()) for i in x])
        y = torch.stack([torch.Tensor(i.tolist()) for i in y])
        dataset = utils.TensorDataset(x, y)
    except ValueError:
        print("Not enough dates to perform time series prediction")
        sys.exit(-1)

    kwargs = {'num_workers': 2, 'pin_memory': True} if opts['use_cuda'] else {}
    return utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
