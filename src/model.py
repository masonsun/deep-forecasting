import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.util import ng_zeros, ng_ones
from options import opts


# define module for time series prediction
# input can be concatenated with exogenous variables
class Predictor(nn.Module):
    def __init__(self, input_size=opts['frame'], exogenous_vars=0):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_size + exogenous_vars, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x


# define module that parameterizes variational distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, frame=opts['frame']):
        super(Encoder, self).__init__()
        # layers
        self.lstm = nn.LSTMCell(frame, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, z_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = F.softplus(self.lstm(x))
        z_mu = self.fc1(hidden)
        z_sigma = torch.exp(self.fc2(hidden))
        # mean vector and positive square root covariance
        # each of size (batch_size x z_dim)
        return z_mu, z_sigma


# define module that parameterizes the observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, frame=opts['frame']):
        super(Decoder, self).__init__()
        # layers
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, frame)

    def forward(self, x):
        # forward computation on latent z
        hidden = F.softplus(self.fc1(x))
        # return the parameter for the output Bernoulli
        # each is of size (batch_size x frame)
        mu_frame = (F.sigmoid(self.fc2(hidden)) + opts['fudge']) * (1-2 * opts['fudge'])
        return mu_frame


# define module for the VAE
class VAE(nn.Module):
    # by default our latent space is 16-dimensional
    def __init__(self, z_dim=16, hidden_dim=32, use_cuda=opts['use_cuda']):
        super(VAE, self).__init__()
        # submodules
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        # parameters
        self.cuda() if use_cuda else None
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.frame = opts['frame']

    # define the model p(x|z)p(z)
    def model(self, x):
        # register decoder with pyro
        pyro.module("decoder", self.decoder)

        # setup hyper-parameters for prior p(z)
        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)

        # sample from prior (value will be sampled by guide when computing the ELBO),
        # decode the latent code z,
        # and score against actual frame
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)
        mu_frame = self.decoder.forward(z)
        pyro.observe("obs", dist.bernoulli, x.view(-1, self.frame), mu_frame)

    # define the guide q(z|x)
    def guide(self, x):
        # register encoder with pyro
        pyro.module("encoder", self.encoder)

        # use the encoder to get the parameters used to define q(z|x),
        # and sample the latent code z
        z_mu, z_sigma = self.encoder.forward(x)
        pyro.sample("latent", dist.normal, z_mu, z_sigma)

    # define a helper function for reconstructing time series
    def reconstruct_ts(self, x):
        # encode, sample in latent space, and decode
        z_mu, z_sigma = self.encoder(x)
        z = dist.normal(z_mu, z_sigma)
        mu_img = self.decoder(z)
        return mu_img

    def model_sample(self, batch_size=1):
        prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
        prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))
        zs = pyro.sample("z", dist.normal, prior_mu, prior_sigma)
        mu = self.decoder.forward(zs)
        xs = pyro.sample("sample", dist.bernoulli, mu)
        return xs, mu
