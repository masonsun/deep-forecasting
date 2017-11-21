import os
import argparse
import pandas as pd

DATA_PATH = '../data/'


def steam_spy(fn):
    # read in data
    data = pd.read_csv(create_fp(fn), sep=',')

    # remove unneeded rows and cols
    data.drop(data.columns[range(-3, 1)], axis=1, inplace=True)
    data.columns = ['id', 'name', 'date', 'users']
    all_rows = data.shape[0]
    data.dropna(inplace=True)
    print("... Deleted {} rows with NaNs\n".format(all_rows - data.shape[0]))

    # format columns
    data.date = pd.to_datetime(data.date, format='%m/%d/%Y %H:%M:%S %p')
    data.date = data.date.dt.date
    data.users = data.users.astype(int)

    # stats
    print("Rows: {}\nCols: {}\nIDs: {}\nNames: {}\nMin date: {}\nMax date: {}\nMin users: {}\nMax users: {}\n".format(
        data.shape[0], data.shape[1],
        len(data.id.unique()), len(data.name.unique()),
        min(data.date), max(data.date),
        min(data.users), max(data.users)))

    # remove redundant column and return data frame
    data.drop(data.columns[1], axis=1)
    return data


def steam_db(fn):
    data = pd.read_csv(create_fp(fn))
    data.drop(data.columns[-2], axis=1, inplace=True)
    data.columns = ['id', 'date', 'users']
    all_rows = data.shape[0]
    data.dropna(inplace=True)

    # format columns
    data.date = pd.to_datetime(data.date, format='%Y-%m-%d %H:%M:%S')
    data.date = data.date.dt.date
    data.id = data.id.astype(int)
    data.users = data.users.astype(int)

    print("... Deleted {} rows with NaNs\n".format(all_rows - data.shape[0]))

    print("Rows: {}\nCols: {}\nIDs: {}\nMin date: {}\nMax date: {}\nMin users: {}\nMax users: {}\n".format(
        data.shape[0], data.shape[1],
        len(data.id.unique()),
        min(data.date), max(data.date),
        min(data.users), max(data.users)))

    return data


def common_processing(data, output_fn='result.csv'):
    # metadata
    print("Number of games: {}\nNumber of days: {}\n".format(
        len(data.id.unique()), 1 + int(str(max(data.date) - min(data.date)).split()[0])))

    # reformat into usable form
    data = pd.pivot_table(data, values='users', index='date', columns=['id'])

    # save to local
    data.to_csv(create_fp(output_fn), sep=',', index=False, header=True)
    print("Saved file to {}".format(create_fp(output_fn)))


def create_fp(fn):
    return os.path.join(DATA_PATH, fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data preprocessing")
    parser.add_argument('-fn', '--filename', type=str, help='name of data file to preprocess')
    args = parser.parse_args()

    curr_files = ['ConsolidatedSteamSpyTimeSeries.csv', 'SteamDB.csv']
    output_files = ['ts_data.csv', 'db_data.csv']

    assert args.filename is not None, 'Please specify a filename'
    assert args.filename in curr_files, 'File specified is currently not available for processing'

    if args.filename == curr_files[0]:
        data = steam_spy(curr_files[0])
        common_processing(data, output_files[0])
    else:
        data = steam_db(curr_files[-1])
        common_processing(data, output_files[-1])

    print("Finished.")
