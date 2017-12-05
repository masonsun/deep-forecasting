import sys
import argparse
import pandas as pd
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

sns.set_style('darkgrid')
sns.set(font_scale=1.5)

text_fs, lab_fs = 40, 28
file_type = 'pdf'

mpl.rcParams['xtick.labelsize'] = lab_fs
mpl.rcParams['ytick.labelsize'] = lab_fs


def x_spread(sequence, num):
    return [sequence[int(np.ceil(i * float(len(sequence)) / num))] for i in range(num)]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--file', default=None)
    args = p.parse_args()
    assert args.file is not None

    # time series
    time_series = pd.read_csv(args.file)
    time_series.date = pd.to_datetime(time_series.date, format='%Y-%m-%d')

    # identifiers
    games_list = pd.read_csv('../data/UsableGamesList.csv', sep=',')
    games_list = games_list.iloc[:, 1:]

    id_to_name = {}
    for _, (m, n) in games_list.iterrows():
        id_to_name[m] = n.title()

    # results
    results = pd.read_csv('../results/results.csv')
    error = pd.read_csv('../results/error.csv')

    # evaluation
    num_games = time_series.shape[1] - 1
    in_dim, out_dim = 365, results.shape[1] - 1
    x_bins = 10
    save = False

    mae = 0
    within_10, within_20, within_interval = 0, 0, 0
    data_pts = 0

    # loop through data
    for i in range(0, num_games):
        game_id = int(time_series.columns[i + 1])
        name = id_to_name[game_id]

        game = time_series.iloc[:, i + 1]
        trimmed_game = np.trim_zeros(game, trim='f')
        trimmed_game = trimmed_game[-(in_dim + out_dim):]

        dates = time_series.date[len(game) - len(trimmed_game):]

        err = scipy.stats.norm.ppf(1 - 0.05) * np.array(error[error.id == game_id].iloc[:, 1:])[0].astype(float)
        res = np.array(results[results.id == game_id].iloc[:, 1:])[0].astype(float)

        # calculate error
        ground_truth = trimmed_game[-out_dim:]
        mae += (np.abs(ground_truth - res)).mean(axis=0) / len(ground_truth)

        data_pts += len(ground_truth)
        within_10 += ((ground_truth - res) / ground_truth < 0.1).sum()
        within_20 += ((ground_truth - res) / ground_truth < 0.2).sum()
        within_interval += np.logical_and(res - err <= ground_truth, ground_truth <= res + err).sum(axis=0)

        # plotting
        fig = plt.figure(figsize=(32, 10))

        ax = fig.add_subplot(111)
        ax.add_patch(patches.Rectangle((dates[-(in_dim + out_dim):].index[0], 0),
                                       in_dim, max(trimmed_game),
                                       ls=':', lw=4,
                                       fill=False, alpha=0.3))

        # ground truth
        plt.scatter(dates.index, trimmed_game, alpha=0.9, label='_nolegend_')
        plt.plot(dates.index, trimmed_game, linewidth=3, alpha=0.9, label='Ground truth', )

        # prediction
        plt.scatter(dates[-out_dim:].index, res, alpha=1, label='_nolegend_')
        plt.plot(dates[-out_dim:].index, res, linewidth=3, alpha=0.9, label='Prediction')

        # uncertainty
        plt.fill_between(dates[-out_dim:].index, res - err, res + err, color='r', alpha=0.3,
                         label="Prediction interval")

        # configuration
        plt.xlabel('Date', fontsize=text_fs, labelpad=20)
        plt.xticks(x_spread(dates.index, x_bins), x_spread(list(dates.dt.strftime('%Y/%m/%d')), x_bins))

        plt.ylabel('Daily concurrent users', fontsize=text_fs, labelpad=30)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.title('Prediction for "{}"'.format(name), fontsize=text_fs, y=1.01)
        plt.tick_params(axis='both', which='major', direction='out', pad=15)
        plt.legend(fontsize=text_fs)
        plt.tight_layout()

        if save:
            plt.savefig('../utils/plots/{}_{}.{}'.format(game_id, ''.join(name.split()), file_type))
        plt.close('all')

        if i + 1 % 50 == 0:
            print("Completed {:.3f}%".format((i + 1) / num_games * 100))

    print('MAE: {:.3f}'.format(mae))
    print('Within 10% of prediction: {:.3f}'.format(within_10 / data_pts))
    print('Within 20% of prediction: {:.3f}'.format(within_20 / data_pts))
    print('Within interval: {:.3f}'.format(within_interval / data_pts))
    sys.exit(0)
