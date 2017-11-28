import argparse
import numpy as np
import pandas as pd
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from src.options import opts
from src.model import Predictor
from utils.utils import save_checkpoint, mse_plot
from utils.load_data import load_data, set_dataloader

# makes stuff reproducible
np.random.seed(1)
torch.manual_seed(12)
torch.cuda.manual_seed(123) if opts['use_cuda'] else None


def data_validation(data, min_len):
    for (g, d) in data:
        if len(d) < min_len:
            raise RuntimeError("Not enough timestamps at id = {}".format(g))


def create_batch(time_series, in_dim, out_dim):
    ts = np.array([t[1] for t in time_series], dtype=int)
    iters = len(ts) - (in_dim + out_dim) + 1

    x = [None] * iters
    y = [None] * iters
    for i in range(iters):
        x[i] = ts[i: i + in_dim]
        y[i] = ts[i + in_dim: i + in_dim + out_dim]
    return set_dataloader(x, y)


def train_nn(train_data, test_data, in_dim, out_dim, verbose=False):
    # losses
    train_losses = []
    test_losses = []

    # globalize
    global model
    global optimizer
    global criterion

    # training
    print("... Training")
    c = 1
    for game_id, time_series in train_data:
        # prepare data
        train_loader = create_batch(time_series, in_dim, out_dim)

        # train over each mini-batch "x" returned by the data loader
        running_loss = 0
        for i, (x, y) in enumerate(train_loader, 0):
            # process inputs
            if opts['use_cuda']:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            optimizer.zero_grad()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()

            running_loss += loss.data[0]

        if verbose:
            gid = '{}...'.format(game_id[:5]) if len(game_id) > 5 else game_id
            print('[Game {}, {:03d}/{:03d}], current training loss: {:.3f}'.format(
                gid, c, len(data), running_loss))
            c += 1

        # training diagnostics
        train_losses.append(running_loss / len(train_loader.dataset))

    # testing
    print("... Testing")
    c = 1
    for game_id, time_series in test_data:
        # prepare data
        test_loader = create_batch(time_series, in_dim, out_dim)

        # test over mini-batch
        running_loss = 0
        for i, (x, y) in enumerate(test_loader, 0):
            # process inputs
            if opts['use_cuda']:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            output = model(x)
            loss = criterion(output, y)
            running_loss += loss.data[0]

        if verbose:
            gid = '{}...'.format(game_id[:5]) if len(game_id) > 5 else game_id
            print('[Game {}, {:03d}/{:03d}], current testing loss: {:.3f}'.format(
                gid, c, len(data), running_loss))
            c += 1

        # testing diagnostics
        test_losses.append(running_loss / len(test_loader.dataset))

    return np.array(train_losses).mean(axis=0), np.array(test_losses).mean(axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train prediction network")
    parser.add_argument('-f', '--filename', type=str, help='data')
    parser.add_argument('-w', '--weights', default=None, help='load trained weights')
    parser.add_argument('-v', '--verbose', default=False, help='verbose mode')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-o', '--output_dim', type=int, default=1, help='number of days to predict')
    parser.add_argument('-m', '--mode', type=int, default=1, help='training: 0, prediction: 1')
    args = parser.parse_args()

    # data
    assert args.filename is not None, 'Please provide a data file'
    data = load_data(args.filename, verbose=args.verbose)
    data = list(data.items())
    data_validation(data, opts['frame'] + args.output_dim)

    # mode
    to_train = False
    if args.mode == 0:
        to_train = True
        print("===== Training =====")
    elif args.mode == 1:
        print("===== Prediction =====")
    else:
        print("Warning: invalid mode, defaulting to prediction.")

    # output dimension
    if args.weights is not None and args.mode != 0:
        states = torch.load(args.weights)
        try:
            if int(states['output_dim']) != args.output_dim:
                raise RuntimeError("Output dim = {}, but was trained with dim = {}.".format(
                    args.output_dim, int(states['output_dim'])))
        except KeyError:
            raise RuntimeError("Incorrect weights loaded.")

    # network
    model = Predictor(model_path=args.weights, output_dim=args.output_dim)
    if opts['use_cuda']:
        model = model.cuda()

    # training mode
    if to_train:
        print("Training network with time frames: {} -> {}".format(opts['frame'], args.output_dim))
        with open('./models/nn_log.txt', 'a') as f:
            f.write('=== New run ===\n')

        # partition into training and testing sets by a 9:1 split
        split = np.ceil(len(data) * 0.9).astype(int)
        train_data = data[:split]
        test_data = data[split:]

        # loss and optimizer
        best_loss = np.inf
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=opts['lr'], betas=opts['beta'], eps=opts['fudge'],
                               weight_decay=opts['w_decay'])

        # history
        train_loss = [None] * args.epochs
        test_loss = [None] * args.epochs

        # loop over data multiple times
        for epoch in range(args.epochs):
            start_time = dt.now()

            # train and test
            train_loss[epoch], test_loss[epoch] = train_nn(train_data=train_data, test_data=test_data,
                                                           in_dim=opts['frame'], out_dim=args.output_dim,
                                                           verbose=args.verbose)

            # logging
            log = '[Epoch {:03d}/{:03d}] Training loss: {:.3f}, Testing loss: {:.3f}, Time: {:.1f}'.format(
                epoch + 1, args.epochs, train_loss[epoch], test_loss[epoch],
                (dt.now() - start_time).total_seconds() / 60)

            with open('./models/nn_log.txt', 'a') as f:
                f.write(log + '\n')

            # checkpoint
            is_best = False
            curr_loss = np.array(test_loss[:epoch + 1]).mean(axis=0)

            if curr_loss < best_loss:
                best_loss = curr_loss
                is_best = True

            if opts['use_cuda']:
                model = model.cpu()

            states = {'epoch': epoch + 1,
                      'state_dict': model.vae.state_dict(),
                      'ff_dict': model.ff.state_dict(),
                      'loss': best_loss,
                      'optimizer': optimizer.state_dict(),
                      'output_dim': args.output_dim}

            save_checkpoint(states, opts['nn_state'], is_best=is_best)

            if opts['use_cuda']:
                model = model.cuda()

        # plot losses over epochs
        mse_plot(np.array(train_loss), np.array(test_loss))

        print("Finished training and testing.")

    # prediction mode
    else:
        print("Prediction with time frames: {} -> {}".format(opts['frame'], args.output_dim))

        results = {}
        for game_id, time_series in data:
            ts = torch.from_numpy(np.array([t[1] for t in time_series], dtype=int)[-opts['frame']:]).float()
            ts = ts.cuda() if opts['use_cuda'] else ts

            next_ts = model(Variable(ts).unsqueeze(0))
            next_ts = next_ts.cpu() if opts['use_cuda'] else next_ts

            results[game_id] = np.round(next_ts.data.numpy().astype(float), 1)[0]

            if args.verbose:
                gid = '{}...'.format(game_id[:5]) if len(game_id) > 5 else game_id
                res = ', '.join(map(str, list(results[game_id])))
                print("Game {} -> {}".format(gid, res))

        assert len(results) == len(data), 'Length mismatch after prediction'
        df = pd.DataFrame(results).transpose().reset_index()
        df.columns = ['id'] + ['t+{}'.format(t + 1) for t in range(args.output_dim)]

        results_filename = 'results.csv'
        print("Saving results to {}".format(results_filename))
        df.to_csv('./{}'.format(results_filename), header=True, index=False)
