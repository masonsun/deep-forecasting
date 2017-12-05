import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
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


def save_predictions(val, filename):
    df = pd.DataFrame(val).transpose().reset_index()
    print(df)
    df.columns = ['id'] + ['t+{}'.format(t + 1) for t in range(df.shape[1] - 1)]
    print("... Saving to {}".format(filename))
    df.to_csv('./results/{}'.format(filename), header=True, index=False)


def create_batch(data, in_dim, out_dim, b_size):
    x, y = [], []
    skipped = 0
    for game_id, time_series in data:
        ts = np.trim_zeros(np.array([t[1] for t in time_series], dtype=int), trim='f')
        if len(ts) < in_dim + out_dim:
            skipped += 1
            if args.verbose:
                print("... Not enough timestamps for {} after trimming: {}".format(game_id, len(ts)))
            continue

        iters = len(ts) - (in_dim + out_dim) + 1
        for i in range(iters):
            x_dim = ts[i: i + in_dim]
            y_dim = ts[i + in_dim: i + in_dim + out_dim]
            assert len(x_dim) == in_dim
            assert len(y_dim) == out_dim
            x.append(x_dim)
            y.append(y_dim)

    assert len(x) == len(y), 'Length mismatch after preparing time series'
    print("... Skipped observations: {}".format(skipped))
    print('... Number of observations: {}'.format(len(x)))
    return set_dataloader(x, y, b_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train prediction network")
    parser.add_argument('-f', '--filename', type=str, help='data')
    parser.add_argument('-w', '--weights', default=None, help='load trained weights')
    parser.add_argument('-m', '--mode', type=int, default=1, help='training: 0, prediction: 1')
    parser.add_argument('-i', '--input_dim', type=int, default=365, help='input dimension')
    parser.add_argument('-o', '--output_dim', type=int, default=90, help='output dimension')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-v', '--verbose', default=False, help='verbose mode')
    parser.add_argument('-t', '--bootstrap', default=10, help='number of bootstrap iterations')
    args = parser.parse_args()

    # dimensions
    in_dim, out_dim = args.input_dim, args.output_dim

    # data
    assert args.filename is not None, 'Please provide a data file'
    data = load_data(args.filename, verbose=args.verbose)
    data = list(data.items())
    data_validation(data, in_dim + out_dim)

    # mode
    to_train = False
    if args.mode == 0:
        to_train = True
        print("===== Training =====")
    elif args.mode == 1:
        print("===== Prediction =====")
    else:
        print("... Warning: invalid mode, defaulting to prediction.")

    # weights
    if args.weights is None and args.mode == 1:
        raise RuntimeError("Predicting without loaded weights")

    # output dimension
    if args.weights is not None and args.mode != 0:
        states = torch.load(args.weights)
        try:
            if int(states['output_dim']) != out_dim:
                raise RuntimeError("Output dim = {}, but was trained with dim = {}.".format(
                    out_dim, int(states['output_dim'])))
        except KeyError:
            raise RuntimeError("Incorrect weights loaded.")

    # network
    model = Predictor(input_dim=in_dim, output_dim=out_dim,
                      batch_size=args.batch_size, model_path=args.weights)

    if opts['use_cuda']:
        model = model.cuda()

    # training mode
    if to_train:
        for param in model.parameters():
            param.requires_grad = True

        print("... Training network with time frames: {} -> {}".format(in_dim, out_dim))
        with open('./models/nn_log.txt', 'a') as f:
            f.write('=== New run ===\n')

        # partition into training and validation sets
        split = np.ceil(len(data) * opts['train_per']).astype(int)
        train_data = data[:split]
        valid_data = data[split:]

        # convert to data loader
        train_loader = create_batch(train_data, in_dim, out_dim, args.batch_size)
        valid_loader = create_batch(valid_data, in_dim, out_dim, args.batch_size)

        # loss and optimizer
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=args.verbose)

        # history
        train_losses = []
        valid_losses = []

        # loop over data multiple times
        best_loss = np.inf
        for e in range(args.epochs):
            start_time = dt.now()

            if args.verbose:
                print("... Training")

            # train over each mini-batch "x" returned by the data loader
            model.train()
            running_loss = 0.0
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

                if args.verbose and i % 5e2 == 0:
                    print(">>> [{:03d}%] current training loss: {:.3f}".format(
                        np.round(100 * (i + 1) * args.batch_size / len(train_loader.dataset)).astype(int),
                        running_loss))

            # training diagnostics
            train_losses.append(running_loss / len(train_loader.dataset))

            if args.verbose:
                print("... Validating")

            # test over mini-batch
            model.eval()
            running_loss = 0
            for i, (x, y) in enumerate(valid_loader, 0):
                # process inputs
                if opts['use_cuda']:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                output = model(x)
                loss = criterion(output, y)
                running_loss += loss.data[0]

                if args.verbose and i % 5e2 == 0:
                    print(">>> [{:03d}%] current validation loss: {:.3f}".format(
                        np.round(100 * (i + 1) * args.batch_size / len(valid_loader.dataset)).astype(int),
                        running_loss))

            # testing diagnostics
            val_loss = running_loss / len(valid_loader.dataset)
            valid_losses.append(val_loss)
            scheduler.step(val_loss)

            # logging
            log = '[Epoch {:03d}/{:03d}] train loss: {:.3f}, val loss: {:.3f}, time: {:.1f}'.format(
                e + 1, args.epochs,
                train_losses[-1], valid_losses[-1],
                (dt.now() - start_time).total_seconds() / 60)
            print('>>> {}'.format(log))

            with open('./models/nn_log.txt', 'a') as f:
                f.write(log + '\n')

            # checkpoint
            is_best = False
            curr_loss = np.array(valid_losses).mean(axis=0)

            if curr_loss < best_loss:
                best_loss = curr_loss
                is_best = True

            if opts['use_cuda']:
                model = model.cpu()

            states = {'epoch': e + 1,
                      'state_dict': model.vae.state_dict(),
                      'ff_dict': model.ff.state_dict(),
                      'loss': best_loss,
                      'optimizer': optimizer.state_dict(),
                      'output_dim': out_dim}

            save_checkpoint(states, filename=opts['nn_state'], dim=out_dim, is_best=is_best)

            if opts['use_cuda']:
                model = model.cuda()

        assert args.epochs == len(train_losses) == len(valid_losses), 'Mismatch across epochs'

        # plot losses across epochs
        mse_plot(np.array(train_losses), np.array(valid_losses))
        print("... Finished training")
        sys.exit(0)

    # prediction mode
    else:
        print("... Prediction with time frames: {} -> {}".format(in_dim, out_dim))

        # to enable MC dropout
        model.train()
        print("... Bootstrapping iterations: {}".format(args.bootstrap))

        result, uncert, actual = {}, {}, {}
        for game_id, time_series in data:
            arr = np.array([t[1] for t in time_series], dtype=int)
            x = torch.from_numpy(arr[-(in_dim + out_dim): -out_dim]).float()
            y = arr[-out_dim:].astype(int)

            x = x.cuda() if opts['use_cuda'] else x
            sample = []

            for b in range(args.bootstrap):
                y_pred = model(Variable(x).unsqueeze(0))
                y_pred = y_pred.cpu() if opts['use_cuda'] else y_pred
                y_pred = np.round(y_pred.data.numpy(), 0).astype(int)[0]
                sample.append(y_pred)

            sample = np.concatenate(sample, axis=0).reshape(args.bootstrap, -1)
            y_sample = sample.mean(axis=0)

            result[game_id] = y_sample
            uncert[game_id] = np.round(np.sqrt(np.sum((sample - y_sample) ** 2, axis=0) / args.bootstrap), 2)
            actual[game_id] = y
            assert result[game_id].shape == uncert[game_id].shape == actual[game_id].shape

            if args.verbose:
                gid = '{}...'.format(game_id[:5]) if len(game_id) > 5 else game_id
                act = ', '.join(map(str, list(actual[game_id])))
                res = ', '.join(map(str, list(result[game_id])))
                err = ', '.join(map(str, list(uncert[game_id])))
                fil = ' ' * (len(gid) + 5)
                print("Game {} --> {}\n{} --> {}\n{} --> {}".format(gid, act, fil, res, fil, err))

        assert len(result) == len(uncert) == len(data), 'Length mismatch after prediction'
        save_predictions(result, 'results.csv')
        save_predictions(uncert, 'error.csv')
        save_predictions(actual, 'ground_truth.csv')
        print("... Finished predicting")
        sys.exit(0)
