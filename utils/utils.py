import shutil
import torch
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set(font_scale=1.5)

FILE_TYPE = 'pdf'


def llk_plot(test_elbo, *args):
    plt.figure(figsize=(30, 10))

    x_lab = 'Epoch'
    y_labs = ['Test ELBO', 'Train ELBO']
    clrs = ['red', 'blue']
    dot_size = 40

    f = lambda x: np.concatenate([np.arange(len(x))[:, sp.newaxis], x[:, sp.newaxis]], axis=1)
    df = pd.DataFrame(data=f(test_elbo), columns=[x_lab, y_labs[0]])

    for a in args:
        assert test_elbo.shape == a.shape
        a = pd.DataFrame(data=f(a), columns=[x_lab, y_labs[1]])
        df = pd.merge(df, a, how='inner')
        break

    g = sns.FacetGrid(df, size=10, aspect=1.5)
    g.map(plt.scatter, x_lab, y_labs[0], label=y_labs[0], s=dot_size, color=clrs[0])
    g.map(plt.plot, x_lab, y_labs[0], label=y_labs[0], color=clrs[0])
    filename = 'test'

    if y_labs[1] in df.columns:
        g.map(plt.scatter, x_lab, y_labs[1], label=y_labs[1], s=dot_size, color=clrs[1])
        g.map(plt.plot, x_lab, y_labs[1], label=y_labs[1], color=clrs[1])
        filename += '_train'

    g.add_legend()
    g.set_axis_labels(x_lab, y_labs[0].split()[-1])
    plt.savefig('./models/{}_elbo.{}'.format(filename, FILE_TYPE))
    plt.close('all')


def mse_plot(train_loss, test_loss):
    assert len(train_loss) == len(test_loss), 'Length mismatch'
    plt.figure(figsize=(30, 10))

    df = pd.DataFrame({
        'Epoch': np.arange(1, len(train_loss) + 1),
        'Training loss': train_loss,
        'Testing loss': test_loss})

    mpl.rcParams.update({'font.size': 18,
                         'xtick.labelsize': 12,
                         'ytick.labelsize': 12})

    ax = df.iloc[:, 1:].plot(title='Prediction network loss')
    ax.set(xlabel='Epoch', ylabel='Loss')
    plt.savefig('./models/nn_loss.{}'.format(FILE_TYPE))
    plt.close('all')


def save_checkpoint(state, filename, is_best=True):
    fp = './models/{}'.format(filename)
    torch.save(state, fp)
    if is_best:
        shutil.copyfile(fp, '{}{}_best.pth'.format(fp.split(filename)[0], filename.split('_')[0]))
