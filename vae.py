import numpy as np
import argparse
import sys
from datetime import datetime as dt

import torch
from torch.autograd import Variable
from pyro.infer import SVI
from pyro.optim import ClippedAdam

from src.options import opts
from src.model import VAE
from utils.utils import llk_plot, save_checkpoint
from utils.load_data import load_data, set_dataloader

# makes stuff reproducible
np.random.seed(1)
torch.manual_seed(12)
torch.cuda.manual_seed(123) if opts['use_cuda'] else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument('-f', '--filename', type=str, help='data')
    parser.add_argument('-w', '--weights', default=None, help='load trained weights')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='epochs')
    parser.add_argument('-d', '--dim', type=int, default=180, help='input/output dimensions')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-v', '--verbose', type=bool, help='verbose mode')
    args = parser.parse_args()

    # tensor dataset
    assert args.filename is not None, 'Please provide a data file'
    print('... Loading dataset')
    data = load_data(args.filename)

    x, y = [], []
    skipped = 0
    for game_id, time_series in data.items():
        ts = np.trim_zeros(np.array([t[1] for t in time_series], dtype=int), trim='f')
        if len(ts) < args.dim * 2:
            skipped += 1
            if args.verbose:
                print("... Not enough timestamps for {} after trimming: {}".format(game_id, len(ts)))
            continue

        iters = len(ts) - 2 * args.dim + 1
        for i in range(iters):
            x.append(ts[i: i + args.dim])
            y.append(ts[i + args.dim: i + 2 * args.dim])

    assert len(x) == len(y), 'Length mismatch after preparing time series'
    print("... Skipped observations: {}".format(skipped))
    print('... Number of observations: {}'.format(len(x)))

    # partition dataset
    split = np.ceil(len(x) * opts['train_per']).astype(int)

    # dataloaders
    print('... Preparing dataloaders')
    start_time = dt.now()
    train_loader = set_dataloader(x[:split], y[:split], args.batch_size)
    test_loader = set_dataloader(x[split:], y[split:], args.batch_size)
    print('... Finished preparing dataloaders: {:.0f}s'.format((dt.now() - start_time).total_seconds()))

    # vae
    print('... Loading model')
    vae = VAE(frame=args.dim, batch_size=args.batch_size)

    # load weights if available
    prev_epoch = 0
    if args.weights is not None:
        try:
            print("Loading trained weights: {}".format(args.weights))
            states = torch.load(args.weights)
            vae.load_state_dict(states['state_dict'])
            prev_epoch = int(states['epoch'])
        except IOError:
            print("File not found: {}".format(args.weights))
            sys.exit(-1)

    # optimizer
    adam_params = {'lr': 1e-6, 'clip_norm': 10, 'weight_decay': opts['w_decay']}
    optimizer = ClippedAdam(adam_params)

    # inference
    svi = SVI(vae.model, vae.guide, optimizer, loss='ELBO')

    # loss
    train_elbo = [None] * args.epochs
    test_elbo = [None] * args.epochs

    # training info
    print("... Training VAE with {} days".format(args.dim))
    with open('./models/vae_log.txt', 'a') as f:
        f.write('=== New run ===\n')

    # training loop
    best_lb = -np.inf
    for epoch, _ in enumerate(range(args.epochs), prev_epoch):
        start_time = dt.now()

        # loss accumulator
        train_loss = 0.
        test_loss = 0.

        # do training over each mini-batch "x" returned by the data loader
        for i, (x, _) in enumerate(train_loader):
            if opts['use_cuda']:
                x = x.cuda()

            # wrap mini-batch in pytorch variable,
            # do ELBO gradient and accumulate loss
            x = Variable(x)
            train_loss += svi.step(x)

            if args.verbose and i % 1e3 == 0:
                print(">>> [{:03d}%] current training ELBO: {:.3f}".format(
                    np.round(100 * (i+1) * args.batch_size / len(train_loader.dataset)).astype(int),
                    train_loss))

        # testing
        for i, (x, _) in enumerate(test_loader):
            if opts['use_cuda']:
                x = x.cuda()

            # wrap mini-batch in pytorch variable,
            # compute ELBO estimate and accumulate loss
            x = Variable(x)
            test_loss += svi.evaluate_loss(x)

            if args.verbose and i % 1e3 == 0:
                print(">>> [{:03d}%] current testing ELBO: {:.3f}".format(
                    np.round(100 * (i+1) * args.batch_size / len(test_loader.dataset)).astype(int),
                    test_loss))

        # record mean training and testing losses
        train_elbo[epoch] = -train_loss / len(train_loader.dataset)
        test_elbo[epoch] = -test_loss / len(test_loader.dataset)

        # logging
        log = '[Epoch {:03d}/{:03d}] Training ELBO: {:.4f}, Testing ELBO: {:.4f}, Mins: {:.1f}'.format(
            epoch + 1, args.epochs, train_elbo[epoch], test_elbo[epoch],
            (dt.now() - start_time).total_seconds() / 60)
        print('>>> {}'.format(log))

        with open('./models/vae_log.txt', 'a') as f:
            f.write(log + '\n')

        # checkpoint
        is_best = False
        curr_lb = np.array(test_elbo[:epoch + 1]).mean(axis=0)

        if curr_lb > best_lb:
            best_lb = curr_lb
            is_best = True

        if opts['use_cuda']:
            vae = vae.cpu()

        states = {'epoch': epoch + 1,
                  'state_dict': vae.state_dict(),
                  'best_lb': best_lb,
                  'optimizer': optimizer.get_state()}

        save_checkpoint(states, filename=opts['vae_state'], dim=args.dim, is_best=is_best)

        if opts['use_cuda']:
            vae = vae.cuda()

    # plot diagnostics
    llk_plot(np.array(test_elbo), np.array(train_elbo))
    llk_plot(np.array(test_elbo))

    print("... Finished")
