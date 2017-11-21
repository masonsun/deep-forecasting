import sys
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.utils.data as utils

from collections import OrderedDict
from src.options import opts

# INFO:
# We should increase batch size if we can definitely say all games have enough data to allow it


def load_data(fp):
    """
    :param fp: relative path to data
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
    print("Dataset:\n{}\n".format(data.head(3)))

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
    x = torch.stack([torch.Tensor(i.tolist()) for i in x])
    y = torch.stack([torch.Tensor(i.tolist()) for i in y])
    dataset = utils.TensorDataset(x, y)

    kwargs = {'num_workers': 1, 'pin_memory': True} if opts['use_cuda'] else {}
    return utils.DataLoader(dataset, batch_size=opts['batch_size'], shuffle=False, **kwargs)
