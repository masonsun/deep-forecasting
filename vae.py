import numpy as np
import argparse
import sys
from datetime import datetime as dt

import torch
from torch.autograd import Variable
from pyro.infer import SVI
from pyro.optim import Adam

from src.options import opts
from src.model import VAE
from utils.utils import llk_plot, save_checkpoint
from utils.load_data import load_data, set_dataloader

# makes stuff reproducible
np.random.seed(1)
torch.manual_seed(12)
torch.cuda.manual_seed(123) if opts['use_cuda'] else None


def train_vae(data, svi, verbose):
    # loss across games
    train_losses = []
    test_losses = []

    # loop through games
    i = 1
    for game_id, time_series in data.items():
        ts = np.array([t[1] for t in time_series], dtype=int)
        iters = len(ts) - 2 * opts['frame'] + 1

        x = [None] * iters
        y = [None] * iters
        for i in range(iters):
            x[i] = ts[i: i + opts['frame']]
            y[i] = ts[i + opts['frame']: i + 2 * opts['frame']]

        # partition dataset
        split = np.ceil(iters * opts['train_per']).astype(int)

        # dataloaders
        train_loader = set_dataloader(x[:split], y[:split])
        test_loader = set_dataloader(x[split:], y[split:])

        # loss accumulator
        train_loss = 0.
        test_loss = 0.

        # do training over each mini-batch "x" returned by the data loader
        for _, (x, _) in enumerate(train_loader):
            # load into cuda memory
            if opts['use_cuda']:
                x = x.cuda()

            # wrap mini-batch in pytorch variable,
            # do ELBO gradient and accumulate loss
            x = Variable(x)
            train_loss += svi.step(x)

        # debugging
        if verbose:
            if len(game_id) > 5:
                game_id = game_id[:5] + '...'
            print('[Game {}, {:03d}/{:03d}], current training ELBO: {:.3f}'.format(
                game_id, i, len(data), train_loss))
            i += 1

        # training diagnostics
        train_losses.append(train_loss / len(train_loader.dataset))

        # testing
        for i, (x, _) in enumerate(test_loader):
            if opts['use_cuda']:
                x = x.cuda()

            # wrap mini-batch in pytorch variable,
            # compute ELBO estimate and accumulate loss
            x = Variable(x)
            test_loss += svi.evaluate_loss(x)

        # testing diagnostics
        test_losses.append(test_loss / len(test_loader.dataset))

    # return mean training and testing losses
    return np.array(train_losses).mean(axis=0), np.array(test_losses).mean(axis=0), svi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument('-f', '--filename', type=str, help='data')
    parser.add_argument('-w', '--weights', default=None, help='load trained weights')
    parser.add_argument('-v', '--verbose', default=False, help='verbose mode')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    args = parser.parse_args()

    assert args.filename is not None, 'Please provide a data file'

    # data
    data = load_data(args.filename)

    # optimizer
    adam_args = {'lr': opts['lr']}
    optimizer = Adam(adam_args)

    # vae
    vae = VAE()

    # load weights if available
    if args.weights is not None:
        try:
            print("Loading trained weights: {}".format(args.weights))
            states = torch.load(args.weights)
            vae.load_state_dict(states['state_dict'])
        except IOError:
            print("File not found: {}".format(args.weights))
            sys.exit(-1)

    # inference
    svi = SVI(vae.model, vae.guide, optimizer, loss='ELBO')

    # loss
    train_elbo = [None] * args.epochs
    test_elbo = [None] * args.epochs

    # training info
    print("Training VAE with time frame = {} days".format(opts['frame']))
    with open('./models/vae_log.txt', 'a') as f:
        f.write('=== New run ===\n')

    # training loop
    best_lb = -np.inf
    for epoch in range(args.epochs):
        start_time = dt.now()

        train_elbo[epoch], test_elbo[epoch], svi = train_vae(data, svi, args.verbose)
        train_elbo[epoch] *= -1
        test_elbo[epoch] *= -1

        # logging
        log = '[Epoch {:03d}/{:03d}] Training ELBO: {:.4f}, Testing ELBO: {:.4f}, Mins: {}'.format(
            epoch + 1, args.epochs, train_elbo[epoch], test_elbo[epoch],
            (dt.now() - start_time).total_seconds() / 60)

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

        save_checkpoint(states, opts['vae_state'], is_best=is_best)

        if opts['use_cuda']:
            vae = vae.cuda()

    # plot diagnostics
    llk_plot(np.array(test_elbo), np.array(train_elbo))
    llk_plot(np.array(test_elbo))

    print("Finished.")
