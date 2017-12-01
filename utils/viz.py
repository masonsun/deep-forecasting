import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
sns.set(font_scale=1.5)

text_fs, lab_fs = 40, 25
mpl.rcParams['xtick.labelsize'] = lab_fs
mpl.rcParams['ytick.labelsize'] = lab_fs

file_type = 'pdf'
save_plot = True

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--filename', default=None)
    args = p.parse_args()
    assert args.filename is not None, 'Please load a file'

    games_list = pd.read_csv('../data/UsableGamesList.csv', sep=',')
    games_list = games_list.iloc[:, 1:]

    id_to_name = {}
    for _, (m, n) in games_list.iterrows():
        id_to_name[m] = n.title()

    time_series = pd.read_csv(args.filename)
    time_series.date = pd.to_datetime(time_series.date, format='%Y-%m-%d')

    num_games = time_series.shape[1] - 1
    for i in range(num_games):
        game_id = int(time_series.columns[i + 1])
        name = id_to_name[game_id]

        game = time_series.iloc[:, i + 1]
        trimmed_game = np.trim_zeros(game, trim='f')

        plt.figure(figsize=(30, 10))
        plt.plot(time_series.date[len(game) - len(trimmed_game):], trimmed_game, label=name)
        plt.legend(fontsize=text_fs)

        plt.xlabel('Date', fontsize=text_fs)
        plt.ylabel('Daily concurrent users', fontsize=text_fs)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.tick_params(axis='both', which='major', direction='out', pad=15)
        plt.tight_layout()

        if i % 20 == 0:
            print("Completed {:.3f}%".format((i + 1) / num_games * 100))

        if save_plot:
            plt.savefig('./plots/{}_{}.{}'.format(game_id, ''.join(name.split()), file_type))
        plt.close('all')
