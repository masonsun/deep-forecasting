# Forecasting using RVAE

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

Actual data cannot be provided due to non-disclosure agreements. However, a dummy dataset ``rvae_temp.csv`` is provided to have a working example. If successful, trained model weights ``vae_best.pth`` will be saved in the ``models`` directory along with a plot of the ELBO across epochs.

You can also pass the flags ``--dim`` and ``--batch_size`` to configure the input/output dimension and batch size, respectively. Default value for both is 1.


### Training the prediction network

After pretraining RVAE, we can train the prediction network as follows:

```Shell
python network.py --filename data/pred_temp.csv --weights models/vae_best.pth --epochs X \
	--input_dim A --output_dim B --mode 0
```

Once again, ``pred_temp.csv`` is only provided to have a working example. The ``--input_dim`` and ``--output_dim`` flags denote the input and output time series dimension (``--input_dim`` should match ``--dim`` or it would be akin to training from scratch). The same batch size flag from ``vae.py`` may also be used here. All default values are 1. When training concludes, the model weights ``nn_best.pth`` will be generated. This can then be used to predict new time series data.

In addition, we define ``--mode`` to be a binary flag that denotes training (0) or prediction/forecasting (1). In prediction mode, ``--bootstrap`` sets the number of forward passes to run (default is 10). This is used for uncertainty estimation.

### Forecasting

Running in prediction mode will also produce two csv files, ``result.csv`` and ``error.csv``, in the ``results`` directory. The first file records the predictions, where each row is a forward pass and each column represents the forecasted values. The second file outputs the sample variance associated with each timestamp. These values can be visualized using ``utils/evaluate.py`` to create a time series plot.