import pandas as pd
import os

DATA_PATH = '../data/'
INPUT_FILENAME = 'ConsolidatedSteamSpyTimeSeries.csv'
OUTPUT_FILENAME = 'ts_data.csv'

# read in data
data = pd.read_csv(os.path.join(DATA_PATH, INPUT_FILENAME), sep=',')

# remove unneeded rows and cols
data.drop(data.columns[range(-3, 1)], axis=1, inplace=True)
data.columns = ['id', 'name', 'date', 'users']
all_rows = data.shape[0]
data.dropna(inplace=True)
print("... Deleted {} rows with NaNs".format(all_rows - data.shape[0]))

# format columns
data.date = pd.to_datetime(data.date, format='%m/%d/%Y %H:%M:%S %p')
data.date = data.date.dt.date
data.users = data.users.astype(int)

# stats
print("Rows: {}\nCols: {}\nIDs: {}\nNames: {}\nMin date: {}\nMax date: {}\nMin users: {}\nMax users: {}".format(
    data.shape[0], data.shape[1],
    len(data.id.unique()), len(data.name.unique()),
    min(data.date), max(data.date),
    min(data.users), max(data.users)))

# sanity check
x = data[data.name == 'Counter-Strike']
print(x.shape[0] == len(x.date.unique()))

# metadata
print("Number of games: {}\nNumber of days: {}".format(
    len(data.id.unique()), 1 + int(str(max(data.date) - min(data.date)).split()[0])))

# reformat into usable form
ts_data = data.drop(data.columns[1], axis=1)
ts_data = pd.pivot_table(data, values='users', index='date', columns=['id'])

# save to local
ts_data.to_csv(os.path.join(DATA_PATH, OUTPUT_FILENAME), sep=',', index=False, header=True)
print("Saved file to {}".format(os.path.join(DATA_PATH, OUTPUT_FILENAME)))
