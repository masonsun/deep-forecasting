# Forecasting using recurrent VAE

A PyTorch/Pyro implementation of a recurrent variational autoencoder (RVAE) for time series forecasting.

### Installation

```Shell
git clone https://github.com/masonsun/deep_forecasting.git
cd ./deep_forecasting
pip install -r requirements.txt
```

Our current implementation only supports Python 3.X. The code will automatically determine whether or not GPUs are available, and will load the network into CUDA memory if possible.


### Training RVAE

To train for X epochs from scratch:
```Shell
python vae.py --filename data/rvae_temp.csv --epochs X
```

To use pretrained weights:
```Shell
python vae.py --filename data/rvae_temp.csv --weights path/to/weights/<weights_file>.pth --epochs X
```

Actual data cannot be provided due to non-disclosure agreements. However, a dummy dataset ``rvae_temp.csv`` is provided to have a working example. If successful, trained model weights will be saved in the ``models`` directory along with a plot of the ELBO across epochs.


### Training the prediction network

After training the RVAE offline, we can train the prediction network as follows:

```Shell
python network.py --filename data/pred_temp.csv --weights models/vae_best.pth --epochs X --output_dim Y
```

Here, Y specifies the number of days into the future we want to predict. The default value is 1.
