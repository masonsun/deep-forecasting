import numpy as np
import shutil
import torch
from torch.autograd import Variable
from pyro.infer import SVI
from pyro.optim import Adam

from src.options import opts
from src.model import VAE
from utils.plots import llk_plot
from utils.load_data import load_data, set_dataloader

# makes stuff reproducible
np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789) if opts['gpu'] else None


def train_vae(data, svi, epoch):
    # loss across games
    train_losses = []
    test_losses = []

    # loop through games
    for game_id, time_series in data.items():
        ts = np.array([t[1] for t in time_series], dtype=int)
        iters = len(ts) - 2 * opts['frame'] + 1

        x = [None] * iters
        y = [None] * iters
        for i in range(iters):
            x[i] = ts[i: i + opts['frame']]
            y[i] = ts[i + opts['frame']: i + 2 * opts['frame']]
        assert None not in x and None not in y

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

        # training diagnostics
        train_losses.append(train_loss / len(train_loader.dataset))

        # testing
        if epoch % opts['test_frequency'] == 0:
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


def save_checkpoint(state, is_best=True, filename=opts['vae_state']):
    fp = '../model/{}'.format(filename)
    torch.save(state, fp)
    if is_best:
        shutil.copyfile(fp, fp.split(filename)[0] + 'model_best.pth')


def main():
    # data
    data = load_data()

    # optimizer
    adam_args = {'lr': opts['lr']}
    optimizer = Adam(adam_args)

    # inference
    vae = VAE()
    svi = SVI(vae.model, vae.guide, optimizer, loss='ELBO')

    # loss
    train_elbo = []
    test_elbo = []

    # training loop
    best_lb = -np.inf
    for epoch in range(opts['epochs']):
        train_elbo[epoch], test_elbo[epoch], svi = train_vae(data, svi)

        print('[Epoch {:03d}] Training loss: {:.5f}, Testing loss: {:.5f}'.format(
            epoch, train_elbo[epoch], test_elbo[epoch]))

        # checkpoint
        curr_lb = np.array(test_elbo).mean(axis=0)
        if curr_lb > best_lb:
            best_lb = curr_lb

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': vae.state_dict(),
                'best_lb': best_lb,
                'optimizer': optimizer.get_state()}, is_best=True)

    # plot diagnostics and return model
    llk_plot(np.array(test_elbo))
    return vae


if __name__ == '__main__':
    model = main()
