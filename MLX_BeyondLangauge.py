"""
Author: Stavros Kassinos (kassinos.stavros@ucy.ac.cy)
Date: September 2024
Code version: 0.0.1

Description:
    This script implements a physics-informed transformer model for simulating heat diffusion in a 2D domain,
    designed to leverage the MLX framework on Apple Silicon.

MIT License

Copyright (c) [2024] [STAVROS KASSINOS]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import prange

numba.config.THREADING_LAYER = 'threadsafe'
import time
import json
import pickle

import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
from functools import partial
import os
import socket

# Set the environment variable
hostname = socket.gethostname()
mx.set_default_device(mx.gpu)

# Set the random seeds for reproducibility
np.random.seed(1)
mx.random.seed(1)

# Configuration parameters
config = {
    "geometry": {
        "rod_length": 1.0,
        "rod_width": 1.0,
        "dx": 0.04,
        "dy": 0.04,
    },
    "boundary_conditions": {
        "left_limits": (0, 1),       # Range of Dirichlet values to be sampled (normalized Temperature)
        "right_limits": (0, 1),
        "top_limits": (0, 1),
        "bottom_limits": (0, 0.1),   # Note smaller range to break symmetry
    },
    "thermal_diffusivity": {
        "alpha_limits": (0.01, 0.1)   # Range of thermal diffusivity values to be sampled (normalized)
    },
    "model_params": {
        "start_predicting_from": 5,  # First few unmasked frames (the "initial" condition)
        "batch_size": 4,             # Number of samples per batch. Watch out for memory requirements
        "epochs": 100,                 # The epoch to finish training at
        "time_steps": 401,           # This corresponds to seq_len
        "num_heads": 16,             # usually 16
        "num_encoder_layers": 12,    # Increase to 24 for Challenge-2
        "mlp_dim": 256,              # 256,
        "embed_dim": 512,            # 512,
        "mask_type": 'block'         # Options: block (whole sequence) or causal (auto-regressive)
    },
    "learning_rate_schedule": {
        2: 1e-5,
        3: 1e-4,
        4: 5e-4,
        5: 1e-3,
        30: 5e-4,
        40: 1e-4
    },
    "run_label": 'run1',                         # Run label used for constructing directory/file names for i/o
    "boundary_segment_strategy": "base_case",    # Options: "base_case", "challenge_1", "challenge_2",
    "training_samples": 12_000,                  # 8400  #12_000 for challenge_1 and 24_000 for challenge_2
    "start_from_scratch": True,      # Flag to indicate if starting from scratch: True = Fresh run
    "checkpoint_epoch":None,         # Set to None if Fresh run or Set to epoch to reload from if start_from_scratch = False
    "current_epoch": None,           # Set to None by default
    "save_checkpoints": False,       # Set to True to save checkpoints. Frequency set by save_interval.
    "compare_current_loaded": False  # The default is False. Set to True if doing a "live reload" during training
}

io_and_plots_config = {
    "plots": {
        "movie_frames": True,    # Set to true to generate frame comparisons
        "num_examples": 20       # Number of test cases to generate frame comparisons for.
    },
    "model_saving": True
}

# Choose the save_interval based on save_checkpoints
if config["save_checkpoints"]:
    config["save_interval"] = 25  # Set a reasonable interval for saving
else:
    config["save_interval"] = config["model_params"]["epochs"] + 1  # Disable saving by setting it beyond the number of epochs

def setup_save_directories(run_name, restart_epoch=None):
    """
    Set up directories for saving model checkpoints, datasets, and other outputs inside the OUTPUTS directory.

    Parameters
    ----------
    run_name : str
        The base name for the current run (e.g., to create a unique directory).
    restart_epoch : int, optional
        The epoch number to append for a restarted run (default is None).

    Returns
    -------
    tuple
        Paths to directories for saving model checkpoints, datasets, frame plots, and inference MSE.
    """
    # Get the directory where the script is located and define the OUTPUTS folder
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "OUTPUTS")
    os.makedirs(output_dir, exist_ok=True)

    if restart_epoch is not None:
        run_name = f"{run_name}_restart_epoch_{restart_epoch}"

    # Define subdirectories within the OUTPUTS directory
    save_dir_path = os.path.join(output_dir, f'Transformer_save_BeyondL_{run_name}')
    dataset_save_dir_path = os.path.join(output_dir, f'Datasets_save_BeyondL_{run_name}')
    frameplots_save_dir_path = os.path.join(output_dir, f'Heatmaps_BeyondL_{run_name}')
    inference_mse_dir_path = os.path.join(output_dir, f'InferenceMSE_BeyondL_{run_name}')

    # Ensure these directories exist
    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(dataset_save_dir_path, exist_ok=True)
    os.makedirs(frameplots_save_dir_path, exist_ok=True)
    os.makedirs(inference_mse_dir_path, exist_ok=True)

    return save_dir_path, dataset_save_dir_path, frameplots_save_dir_path, inference_mse_dir_path



def setup_load_directories(run_name, checkpoint_epoch):
    """
    Set up directories for loading model checkpoints and datasets from the OUTPUTS directory.

    Parameters
    ----------
    run_name : str
        The base name for the current run (e.g., to create a unique directory).
    checkpoint_epoch : int
        The epoch number to load the checkpoint from.

    Returns
    -------
    tuple
        Paths to directories for loading model checkpoints and datasets.
    """
    # Get the directory where the script is located and define the OUTPUTS folder
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "OUTPUTS")

    # Define subdirectories within the OUTPUTS directory
    load_dir_path = os.path.join(output_dir, f'Transformer_save_BeyondL_{run_name}')
    dataset_load_dir_path = os.path.join(output_dir, f'Datasets_save_BeyondL_{run_name}')

    # Optionally, you can check if these directories exist and raise an error if not found
    if not os.path.exists(load_dir_path):
        raise FileNotFoundError(f"Checkpoint directory {load_dir_path} does not exist!")
    if not os.path.exists(dataset_load_dir_path):
        raise FileNotFoundError(f"Dataset directory {dataset_load_dir_path} does not exist!")

    return load_dir_path, dataset_load_dir_path


# Set the run name as a parameter using Config parameters for identification of files/folders
run_name = str(config["run_label"]) + "_" + str(config["boundary_segment_strategy"]) + "_" + str(config["model_params"]["mask_type"])
# Set the run name based on configuration

if not config["start_from_scratch"]:
    # Load model and optimizer from a checkpoint directory
    load_dir_path, dataset_load_dir_path = setup_load_directories(run_name, config["checkpoint_epoch"])

# Now set up new directories for saving after restarting
save_dir_path, dataset_save_dir_path, frameplots_save_dir_path, inference_mse_dir_path = setup_save_directories(run_name, config["checkpoint_epoch"])


# Define the directory path where the model and configuration will be saved
model_base_file_name = f'heat_diffusion_2D_model_BeyondL_{run_name}'
hyper_base_file_name = f'config.json_BeyondL_{run_name}'
optimizer_base_file_name = f'optimizer_state_BeyondL_{run_name}'


def save_datasets(train_data, train_alphas, train_dts, val_data, val_alphas, val_dts, test_data, test_alphas, test_dts,
                  dir_path):
    """
    Saves the training, validation, and test datasets to the specified directory.

    This function stores the training, validation, and test datasets, along with their respective thermal
    diffusivities and time steps, into the provided directory path. The data is saved in `.npy` format using
    the MLX `mx.save` utility.

    Parameters
    ----------
    train_data : mlx.core.array
        Training dataset representing the temperature distribution field over time.
    train_alphas : mlx.core.array
        Thermal diffusivity values for the training dataset.
    train_dts : mlx.core.array
        Time steps for the training dataset.
    val_data : mlx.core.array
        Validation dataset representing the temperature distribution field over time.
    val_alphas : mlx.core.array
        Thermal diffusivity values for the validation dataset.
    val_dts : mlx.core.array
        Time steps for the validation dataset.
    test_data : mlx.core.array
        Test dataset representing the temperature distribution field over time.
    test_alphas : mlx.core.array
        Thermal diffusivity values for the test dataset.
    test_dts : mlx.core.array
        Time steps for the test dataset.
    dir_path : str
        Directory path where the datasets will be saved.

    Returns
    -------
    None
        The datasets are saved to the specified directory.
    """
    os.makedirs(dir_path, exist_ok=True)
    mx.save(os.path.join(dir_path, 'train_data'), train_data)
    mx.save(os.path.join(dir_path, 'train_alphas'), train_alphas)
    mx.save(os.path.join(dir_path, 'train_dts'), train_dts)
    mx.save(os.path.join(dir_path, 'val_data'), val_data)
    mx.save(os.path.join(dir_path, 'val_alphas'), val_alphas)
    mx.save(os.path.join(dir_path, 'val_dts'), val_dts)
    mx.save(os.path.join(dir_path, 'test_data'), test_data)
    mx.save(os.path.join(dir_path, 'test_alphas'), test_alphas)
    mx.save(os.path.join(dir_path, 'test_dts'), test_dts)


def load_datasets(dir_path):
    """
    Loads the training, validation, and test datasets from the specified directory.

    This function retrieves previously saved training, validation, and test datasets, along with their thermal
    diffusivities and time steps, from `.npy` files. The data is loaded using the MLX `mx.load` utility.

    Parameters
    ----------
    dir_path : str
        Directory path where the datasets are stored.

    Returns
    -------
    tuple of (mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array)
        - train_data : mlx.core.array
            Training dataset representing the temperature distribution field over time.
        - train_alphas : mlx.core.array
            Thermal diffusivity values for the training dataset.
        - train_dts : mlx.core.array
            Time steps for the training dataset.
        - val_data : mlx.core.array
            Validation dataset representing the temperature distribution field over time.
        - val_alphas : mlx.core.array
            Thermal diffusivity values for the validation dataset.
        - val_dts : mlx.core.array
            Time steps for the validation dataset.
        - test_data : mlx.core.array
            Test dataset representing the temperature distribution field over time.
        - test_alphas : mlx.core.array
            Thermal diffusivity values for the test dataset.
        - test_dts : mlx.core.array
            Time steps for the test dataset.
    """
    train_data = mx.load(os.path.join(dir_path, 'train_data.npy'))
    train_alphas = mx.load(os.path.join(dir_path, 'train_alphas.npy'))
    train_dts = mx.load(os.path.join(dir_path, 'train_dts.npy'))
    val_data = mx.load(os.path.join(dir_path, 'val_data.npy'))
    val_alphas = mx.load(os.path.join(dir_path, 'val_alphas.npy'))
    val_dts = mx.load(os.path.join(dir_path, 'val_dts.npy'))
    test_data = mx.load(os.path.join(dir_path, 'test_data.npy'))
    test_alphas = mx.load(os.path.join(dir_path, 'test_alphas.npy'))
    test_dts = mx.load(os.path.join(dir_path, 'test_dts.npy'))
    return train_data, train_alphas, train_dts, val_data, val_alphas, val_dts, test_data, test_alphas, test_dts


def initialize_geometry_and_bcs(config):
    """
    Initializes the geometry and boundary conditions for the 2D temperature distribution simulation.

    This function calculates the number of grid points in the x and y directions (nx, ny) based on the
    specified rod length, width, and spatial steps. It also generates boundary conditions and thermal diffusivity
    values for training, validation, and testing by splitting them according to the provided limits in the configuration.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration for the geometry, boundary conditions, and thermal diffusivity.
        Example structure:
        config = {
            "geometry": {
                "rod_length": float,  # Length of the rod
                "rod_width": float,   # Width of the rod
                "dx": float,          # Spatial step in the x direction
                "dy": float           # Spatial step in the y direction
            },
            "boundary_conditions": {
                "left_limits": tuple,  # Boundary conditions on the left side
                "right_limits": tuple, # Boundary conditions on the right side
                "top_limits": tuple,   # Boundary conditions on the top side
                "bottom_limits": tuple # Boundary conditions on the bottom side
            },
            "thermal_diffusivity": {
                "alpha_limits": tuple  # Limits for the thermal diffusivity values
            },
            "training_samples": int  # Number of samples to generate
        }

    Returns
    -------
    tuple
        A tuple containing:
        - nx : int
            Number of grid points along the x-axis.
        - ny : int
            Number of grid points along the y-axis.
        - training_bcs : list
            Boundary conditions for the training dataset.
        - validation_bcs : list
            Boundary conditions for the validation dataset.
        - test_bcs : list
            Boundary conditions for the test dataset.
        - training_alphas : list or numpy.ndarray
            Thermal diffusivity values for the training dataset.
        - validation_alphas : list or numpy.ndarray
            Thermal diffusivity values for the validation dataset.
        - test_alphas : list or numpy.ndarray
            Thermal diffusivity values for the test dataset.
    """
    geom = config["geometry"]
    bcs = config["boundary_conditions"]
    alphas = config["thermal_diffusivity"]

    # Calculate derived parameters
    nx = int(geom["rod_length"] / geom["dx"]) + 1
    ny = int(geom["rod_width"] / geom["dy"]) + 1

    # Generate boundary conditions
    training_bcs, validation_bcs, test_bcs, training_alphas, validation_alphas, test_alphas = \
        generate_bcs_and_split_2D(
            config["training_samples"],
            bcs["left_limits"], bcs["right_limits"],
            bcs["top_limits"], bcs["bottom_limits"],
            alphas["alpha_limits"]
        )

    return nx, ny, training_bcs, validation_bcs, test_bcs, training_alphas, validation_alphas, test_alphas


def generate_datasets(config, training_bcs, validation_bcs, test_bcs, training_alphas,
                      validation_alphas, test_alphas):
    """
    Generates training, validation, and test datasets for 2D temperature distribution simulations
    using the provided boundary conditions and thermal diffusivities.

    This function uses the geometry and model parameters from the configuration to generate
    datasets for training, validation, and testing. Each dataset represents the temperature
    distribution field over time for a 2D grid, generated using the provided boundary conditions
    and thermal diffusivities.

    Parameters
    ----------
    config : dict
        Dictionary containing the model configuration, including the geometry and model parameters.
        Example structure:
        config = {
            "geometry": {
                "rod_length": float,  # Length of the rod in the x direction
                "rod_width": float,   # Width of the rod in the y direction
                "dx": float,          # Spatial step in the x direction
                "dy": float           # Spatial step in the y direction
            },
            "model_params": {
                "time_steps": int     # Number of time steps for the simulation
            }
        }
    training_bcs : tuple
        Boundary conditions for the training dataset, provided as tuples for each side of the 2D grid
        (e.g., temperature or flux values at the boundaries).
    validation_bcs : tuple
        Boundary conditions for the validation dataset.
    test_bcs : tuple
        Boundary conditions for the test dataset.
    training_alphas : list or numpy.ndarray
        Thermal diffusivities (alpha values) for the training dataset.
    validation_alphas : list or numpy.ndarray
        Thermal diffusivities (alpha values) for the validation dataset.
    test_alphas : list or numpy.ndarray
        Thermal diffusivities (alpha values) for the test dataset.

    Returns
    -------
    tuple of (mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array)
        A tuple containing arrays for training, validation, and test datasets:
        - training_data : mlx.core.array
            Generated training dataset representing the temperature distribution field over time.
        - training_alphas : mlx.core.array
            Thermal diffusivities for the training dataset.
        - training_dts : mlx.core.array
            Time step sizes for the training dataset.
        - validation_data : mlx.core.array
            Generated validation dataset representing the temperature distribution field over time.
        - validation_alphas : mlx.core.array
            Thermal diffusivities for the validation dataset.
        - validation_dts : mlx.core.array
            Time step sizes for the validation dataset.
        - test_data : mlx.core.array
            Generated test dataset representing the temperature distribution field over time.
        - test_alphas : mlx.core.array
            Thermal diffusivities for the test dataset.
        - test_dts : mlx.core.array
            Time step sizes for the test dataset.
    """
    geom = config["geometry"]
    model_params = config["model_params"]
    boundary_segment_strategy = config["boundary_segment_strategy"]

    # Generate datasets using the provided utility function
    training_data, training_alphas, training_dts = generate_heat_data_2D(
        geom["rod_length"], geom["rod_width"],
        geom["dx"], geom["dy"],
        model_params["time_steps"],
        *training_bcs, training_alphas, boundary_segment_strategy
    )

    validation_data, validation_alphas, validation_dts = generate_heat_data_2D(
        geom["rod_length"], geom["rod_width"],
        geom["dx"], geom["dy"],
        model_params["time_steps"],
        *validation_bcs, validation_alphas, boundary_segment_strategy
    )

    test_data, test_alphas, test_dts = generate_heat_data_2D(
        geom["rod_length"], geom["rod_width"],
        geom["dx"], geom["dy"],
        model_params["time_steps"],
        *test_bcs, test_alphas, boundary_segment_strategy
    )

    return mx.array(training_data), mx.array(training_alphas), mx.array(training_dts), \
        mx.array(validation_data), mx.array(validation_alphas), mx.array(validation_dts), \
        mx.array(test_data), mx.array(test_alphas), mx.array(test_dts)


def initialize_model_and_optimizer(config, nx, ny):
    """
    Initializes the heat diffusion model and its optimizer based on the provided configuration.

    This function creates an instance of the `HeatDiffusionModel` using the grid dimensions (`nx`, `ny`)
    and other model parameters from the configuration. It also initializes an optimizer, specifically Adam,
    for training the model.

    Parameters
    ----------
    config : dict
        Dictionary containing the model configuration, including parameters related to the number of heads,
        layers, MLP dimensions, embedding dimensions, and other hyperparameters.
        Example structure:
        config = {
            "model_params": {
                "time_steps": int,           # Number of time steps for the simulation
                "num_heads": int,            # Number of attention heads
                "num_encoder_layers": int,   # Number of encoder layers
                "mlp_dim": int,              # Dimensionality of the MLP layer
                "embed_dim": int,            # Embedding dimensionality
                "start_predicting_from": int,# Time step from which predictions start
                "mask_type": str             # Mask type used in the model
            }
        }
    nx : int
        Number of grid points along the x-axis (calculated based on the geometry).
    ny : int
        Number of grid points along the y-axis (calculated based on the geometry).

    Returns
    -------
    tuple
        A tuple containing:
        - model : HeatDiffusionModel
            An instance of the `HeatDiffusionModel` initialized with the specified parameters.
        - optimizer : mlx.optimizers.Adam
            An instance of the Adam optimizer initialized with a learning rate of 0.0. The learning rate can
            be adjusted later during training.

    """
    model_params = config["model_params"]

    model = HeatDiffusionModel(
        ny, nx,
        model_params["time_steps"], model_params["num_heads"],
        model_params["num_encoder_layers"], model_params["mlp_dim"],
        model_params["embed_dim"], model_params["start_predicting_from"],
        model_params["mask_type"]
    )
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=0.0)

    return model, optimizer


class HeatDiffusionModel(nn.Module):
    """
    Transformer-based model for predicting 2D temperature distributions over time.

    This model uses sinusoidal positional encodings for spatial and temporal dimensions and a transformer
    encoder for learning the time evolution of the temperature distribution. It incorporates masking techniques
    (causal or block) to handle sequence prediction tasks for temperature distributions, taking into account
    the thermal diffusivity of the material.

    Parameters
    ----------
    ny : int
        Number of grid points along the y-axis (height).
    nx : int
        Number of grid points along the x-axis (width).
    seq_len : int
        Sequence length (number of time steps) for the prediction.
    num_heads : int
        Number of attention heads in the transformer encoder.
    num_encoder_layers : int
        Number of layers in the transformer encoder.
    mlp_dim : int
        Dimensionality of the MLP (feed-forward) layers inside the transformer encoder.
    embed_dim : int
        Dimensionality of the embedding space for the input data.
    start_predicting_from : int
        Time step from which predictions should start (for masking).
    mask_type : str
        Type of mask to apply, either 'causal' (for autoregressive modeling) or 'block' (for more structured
        sequence masking).

    Methods
    -------
    create_src_block_mask(seq_len)
        Creates a block mask for the input sequence with unmasked time steps starting from `start_predicting_from`.
    create_src_causal_mask(seq_len)
        Creates a causal mask for the input sequence, ensuring that each time step can only attend to past and
        current time steps.
    spatial_positional_encoding(ny, nx)
        Generates sinusoidal positional encodings for the spatial dimensions (y and x).
    temporal_positional_encoding(seq_len, batch_size)
        Generates sinusoidal positional encodings for the temporal dimension (time steps).
    generate_sinusoidal_encoding(length, embed_dim)
        Generates sinusoidal positional encodings given the length of the sequence and the embedding dimension.

    Returns
    -------
    output : mlx.core.array
        Predicted temperature distribution over time for the input sequence.
    """

    def __init__(self, ny, nx, seq_len, num_heads, num_encoder_layers,
                 mlp_dim, embed_dim, start_predicting_from, mask_type):
        super().__init__()
        self.seq_len = seq_len
        self.output_seq_len = seq_len
        self.ny = ny
        self.nx = nx
        self.input_dim = ny * nx
        self.embed_dim = embed_dim
        self.spatial_features = embed_dim // 2
        self._start_predicting_from = start_predicting_from
        self._mask_type = mask_type

        self.projection_spatial_enc = nn.Linear(
            ny * nx * self.spatial_features, self.embed_dim)

        self.positional_encoding_y = nn.SinusoidalPositionalEncoding(
            dims=self.spatial_features, max_freq=1, cos_first=False,
            scale=(1. / (np.sqrt(self.spatial_features // 2))), full_turns=False)
        self.positional_encoding_x = nn.SinusoidalPositionalEncoding(
            dims=self.spatial_features, max_freq=1, cos_first=False,
            scale=(1. / (np.sqrt(self.spatial_features // 2))), full_turns=False)
        self.positional_encoding_t = nn.SinusoidalPositionalEncoding(
            dims=self.embed_dim, max_freq=1,
            scale=(1. / (np.sqrt(self.embed_dim // 2))), full_turns=False)

        self.transformer_encoder = nn.TransformerEncoder(
            num_layers=num_encoder_layers, dims=embed_dim, num_heads=num_heads,
            mlp_dims=mlp_dim, checkpoint=False)

        self.output_projection = nn.Linear(embed_dim, ny * nx)

        self.diffusivity_embedding = nn.Linear(1, embed_dim)

        self.layer_normalizer = nn.LayerNorm(dims=embed_dim)

        if self._mask_type == 'causal':
            self.mask = self.create_src_causal_mask(self.seq_len)
        elif self._mask_type == 'block':
            self.mask = self.create_src_block_mask(self.seq_len)
        else:
            raise ValueError("Unsupported mask type")

    def create_src_block_mask(self, seq_len):
        mask = mx.full((seq_len, seq_len), -mx.inf, dtype=mx.float32)
        mask[:, :self._start_predicting_from] = 0
        return mask

    def create_src_causal_mask(self, seq_len):
        mask = mx.triu(-mx.inf * mx.ones((seq_len, seq_len)), k=0)
        mask[:, :self._start_predicting_from] = 0
        return mask

    def create_tgt_causal_mask(self, seq_len):
        mask = mx.triu(-mx.inf * mx.ones((seq_len, seq_len)), k=0)
        mask[:, :self._start_predicting_from] = 0
        return mask

    def __call__(self, src, alpha):

        batch_size, seq_len, _, _ = src.shape
        # print(f'seq_len: {seq_len}')
        src_unflattened = src[:, :, :]
        src_expanded = mx.expand_dims(src_unflattened, -1)
        pos_enc_ny, pos_enc_nx = self.spatial_positional_encoding(self.ny, self.nx)
        src_pos_enc_y = src_expanded + pos_enc_ny
        src_pos_enc = src_pos_enc_y + pos_enc_nx
        src_pos_enc_flattened = src_pos_enc[:, :, :, :].reshape(-1, seq_len,
                                                                self.ny * self.nx * self.spatial_features)
        src_projected = self.projection_spatial_enc(src_pos_enc_flattened)

        temporal_enc = self.temporal_positional_encoding(seq_len, batch_size)

        src_encoded = src_projected + temporal_enc

        alpha_reshaped = alpha.reshape(-1, 1)
        alpha_embed = self.diffusivity_embedding(alpha_reshaped)

        alpha_embed_expanded = mx.expand_dims(alpha_embed, axis=1)
        alpha_embed_expanded = mx.broadcast_to(alpha_embed_expanded,
                                               (batch_size, seq_len, self.embed_dim))

        src_encoded += alpha_embed_expanded

        encoded = self.transformer_encoder(src_encoded, mask=self.mask)

        normalized = self.layer_normalizer(encoded)

        output = self.output_projection(normalized)

        output = output.reshape(batch_size, self.output_seq_len, self.ny, self.nx)

        return output

    def spatial_positional_encoding(self, ny, nx):
        nx_encoding = mx.expand_dims(mx.expand_dims(mx.expand_dims(
            self.positional_encoding_x(mx.arange(self.nx)), 0), 0), 1)
        ny_encoding = mx.expand_dims(mx.expand_dims(mx.expand_dims(
            self.positional_encoding_y(mx.arange(self.ny)), 0), 0), 3)
        return ny_encoding, nx_encoding

    def temporal_positional_encoding(self, seq_len, batch_size):
        # temporal_encoding = mx.expand_dims(self.generate_sinusoidal_encoding(seq_len, self.embed_dim), axis=0)

        # Option 2: use built-in temporal positional encoding
        temporal_encoding = mx.expand_dims(self.positional_encoding_t(mx.arange(self.seq_len)), axis=0)

        return temporal_encoding

    def generate_sinusoidal_encoding(self, length, embed_dim):
        position = mx.arange(0, length, dtype=mx.float32)
        position = mx.expand_dims(position, axis=1)
        div_term = mx.exp(mx.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        encoding = mx.zeros((length, embed_dim))
        encoding[:, 0::2] = mx.sin(position * div_term)
        encoding[:, 1::2] = mx.cos(position * div_term)
        return encoding


def generate_bcs_and_split_2D(num_samples, left_limits, right_limits, top_limits, bottom_limits,
                              alpha_limits, synchronized_shuffling=False):
    """
    Generates boundary conditions and thermal diffusivity values for a 2D heat diffusion simulation,
    and splits them into training, validation, and test sets.

    Args:
        num_samples (int): The total number of samples to generate.
        left_limits (tuple): A tuple specifying the minimum and maximum values for the left boundary condition (min, max).
        right_limits (tuple): A tuple specifying the minimum and maximum values for the right boundary condition (min, max).
        top_limits (tuple): A tuple specifying the minimum and maximum values for the top boundary condition (min, max).
        bottom_limits (tuple): A tuple specifying the minimum and maximum values for the bottom boundary condition (min, max).
        alpha_limits (tuple): A tuple specifying the minimum and maximum values for the thermal diffusivity (min, max).
        synchronized_shuffling (bool, optional): If True, shuffles all boundary conditions and alphas together,
                                                 keeping them in sync. If False, shuffles each independently. Defaults to False.

    Returns:
        training_bcs (tuple): A tuple of arrays representing boundary conditions for the training set (left, right, top, bottom).
        validation_bcs (tuple): A tuple of arrays representing boundary conditions for the validation set (left, right, top, bottom).
        test_bcs (tuple): A tuple of arrays representing boundary conditions for the test set (left, right, top, bottom).
        training_alphas (np.ndarray): Array of thermal diffusivity values for the training set.
        validation_alphas (np.ndarray): Array of thermal diffusivity values for the validation set.
        test_alphas (np.ndarray): Array of thermal diffusivity values for the test set.
    """
    left_bcs = np.linspace(left_limits[0], left_limits[1], num_samples)
    right_bcs = np.linspace(right_limits[0], right_limits[1], num_samples)
    top_bcs = np.linspace(top_limits[0], top_limits[1], num_samples)
    bottom_bcs = np.linspace(bottom_limits[0], bottom_limits[1], num_samples)
    alphas = np.linspace(alpha_limits[0], alpha_limits[1], num_samples)

    if synchronized_shuffling:
        combined = np.array(list(zip(left_bcs, right_bcs, top_bcs, bottom_bcs, alphas)))
        np.random.shuffle(combined)
        left_bcs, right_bcs, top_bcs, bottom_bcs, alphas = zip(*combined)
        left_bcs = np.array(left_bcs)
        right_bcs = np.array(right_bcs)
        top_bcs = np.array(top_bcs)
        bottom_bcs = np.array(bottom_bcs)
        alphas = np.array(alphas)
    else:
        np.random.shuffle(left_bcs)
        np.random.shuffle(right_bcs)
        np.random.shuffle(top_bcs)
        np.random.shuffle(bottom_bcs)
        np.random.shuffle(alphas)

    training_left = left_bcs[:int(0.7 * num_samples)]
    validation_left = left_bcs[int(0.7 * num_samples):int(0.9 * num_samples)]
    test_left = left_bcs[int(0.9 * num_samples):]

    training_right = right_bcs[:int(0.7 * num_samples)]
    validation_right = right_bcs[int(0.7 * num_samples):int(0.9 * num_samples)]
    test_right = right_bcs[int(0.9 * num_samples):]

    training_top = top_bcs[:int(0.7 * num_samples)]
    validation_top = top_bcs[int(0.7 * num_samples):int(0.9 * num_samples)]
    test_top = top_bcs[int(0.9 * num_samples):]

    training_bottom = bottom_bcs[:int(0.7 * num_samples)]
    validation_bottom = bottom_bcs[int(0.7 * num_samples):int(0.9 * num_samples)]
    test_bottom = bottom_bcs[int(0.9 * num_samples):]

    training_alphas = alphas[:int(0.7 * num_samples)]
    validation_alphas = alphas[int(0.7 * num_samples):int(0.9 * num_samples)]
    test_alphas = alphas[int(0.9 * num_samples):]

    return (training_left, training_right, training_top, training_bottom), \
        (validation_left, validation_right, validation_top, validation_bottom), \
        (test_left, test_right, test_top, test_bottom), \
        training_alphas, validation_alphas, test_alphas


@numba.njit
def generate_heat_data_2D(rod_length, rod_width, dx, dy, time_steps,
                          left_bcs, right_bcs, top_bcs, bottom_bcs, alphas, boundary_segment_strategy="base_case"):
    """
    Generates the temperature distribution field data for the heat equation in a 2D rod over time.

    Parameters
    ----------
    rod_length : float
        Length of the 2D rod along the x-axis.
    rod_width : float
        Width of the 2D rod along the y-axis.
    dx : float
        Spatial step size along the x-axis.
    dy : float
        Spatial step size along the y-axis.
    time_steps : int
        Number of time steps to simulate.
    left_bcs : list or numpy.ndarray
        Boundary condition values for the left side of the rod for each sample.
    right_bcs : list or numpy.ndarray
        Boundary condition values for the right side of the rod for each sample.
    top_bcs : list or numpy.ndarray
        Boundary condition values for the top side of the rod for each sample.
    bottom_bcs : list or numpy.ndarray
        Boundary condition values for the bottom side of the rod for each sample.
    alphas : list or numpy.ndarray
        Thermal diffusivity values for each sample.
    boundary_segment_strategy : str
        Strategy for applying boundary segments. Options are "base_case", "challenge_1", and "challenge_2".

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        solutions, solutions_alphas, solutions_dts
    """
    num_samples = len(left_bcs)
    nx = int(rod_length / dx) + 1
    ny = int(rod_width / dy) + 1

    solutions = np.zeros((num_samples, time_steps, ny, nx))
    solutions_alphas = np.zeros(num_samples)
    solutions_dts = np.zeros(num_samples)

    for n in prange(num_samples):
        dt = dx ** 2 / (10 * alphas[n])
        T = np.full((ny, nx), np.random.uniform(0.0, 1.0))

        T[:, 0] = left_bcs[n]
        T[:, -1] = right_bcs[n]
        T[0, :] = top_bcs[n]
        T[-1, :] = bottom_bcs[n]

        if boundary_segment_strategy == "challenge_1":
            # Fixed boundary segments on specific sides
            side1, side2 = 0, 1
            pos1, pos2 = 8, 8  # Set positions for segments
            apply_fixed_segments(T, side1, side2, pos2, pos2)

        elif boundary_segment_strategy == "challenge_2":
            # Randomly place boundary segments
            side1, side2 = np.random.choice(4, 2, replace=False)
            pos1 = np.random.randint(0, ny - 4 if side1 < 2 else nx - 4)
            pos2 = np.random.randint(0, ny - 4 if side2 < 2 else nx - 4)
            apply_random_segments(T, side1, side2, pos1, pos2)

        # Store the initial condition
        solutions[n, 0] = T

        for t in range(1, time_steps):
            T_new = T.copy()
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    T_new[j, i] = T[j, i] + alphas[n] * dt / dx ** 2 * (
                            T[j, i + 1] - 2 * T[j, i] + T[j, i - 1]
                    ) + alphas[n] * dt / dy ** 2 * (
                                          T[j + 1, i] - 2 * T[j, i] + T[j - 1, i]
                                  )

            # Apply the boundary conditions again
            T_new[:, 0] = left_bcs[n]
            T_new[:, -1] = right_bcs[n]
            T_new[0, :] = top_bcs[n]
            T_new[-1, :] = bottom_bcs[n]

            # Reapply the boundary segments based on the strategy (if in Challenge configuration)
            if boundary_segment_strategy == "challenge_1":
                apply_fixed_segments(T_new, side1, side2, pos1, pos2)
            elif boundary_segment_strategy == "challenge_2":
                apply_random_segments(T_new, side1, side2, pos1, pos2)

            solutions[n, t] = T_new
            T = T_new

        solutions_dts[n] = dt
        solutions_alphas[n] = alphas[n]

    return solutions, solutions_alphas, solutions_dts

@numba.njit
def apply_fixed_segments(T, side1, side2, pos1, pos2):
    """Apply fixed boundary segments to the temperature field."""
    if side1 == 0:  # Left side
        T[pos1:pos1 + 4, 0] = 1.0  # Apply segment on the left
    if side2 == 1:  # Right side
        T[pos2:pos2 + 4, -1] = 0.0  # Apply segment on the right

@numba.njit
def apply_random_segments(T, side1, side2, pos1, pos2):
    """Randomly place boundary segments on the temperature field."""
    if side1 == 0:  # Left
        T[pos1:pos1 + 4, 0] = 1.0
    elif side1 == 1:  # Right
        T[pos1:pos1 + 4, -1] = 1.0
    elif side1 == 2:  # Top
        T[0, pos1:pos1 + 4] = 1.0
    elif side1 == 3:  # Bottom
        T[-1, pos1:pos1 + 4] = 1.0

    if side2 == 0:  # Left
        T[pos2:pos2 + 4, 0] = 0.0
    elif side2 == 1:  # Right
        T[pos2:pos2 + 4, -1] = 0.0
    elif side2 == 2:  # Top
        T[0, pos2:pos2 + 4] = 0.0
    elif side2 == 3:  # Bottom
        T[-1, pos2:pos2 + 4] = 0.0


def data_loader_2D(data, alphas, solution_dts, batch_size, shuffle=True):
    """
    Data loader function to create mini-batches from the generated temperature distribution data for training.

    This function takes the generated temperature distribution data and splits it into mini-batches.
    It yields the source data (inputs), target data (outputs), thermal diffusivities (alphas),
    and time steps (dts) for each batch. The data can optionally be shuffled to ensure randomness in
    the mini-batches during training.

    Parameters
    ----------
    data : numpy.ndarray or mlx.core.array
        The 4D array representing the generated temperature distribution data, with shape
        (num_samples, time_steps, ny, nx), where:
        - num_samples is the number of samples,
        - time_steps is the number of time steps in the simulation,
        - ny is the number of grid points along the y-axis,
        - nx is the number of grid points along the x-axis.
    alphas : numpy.ndarray or mlx.core.array
        1D array of thermal diffusivity values (alphas) for each sample, with shape (num_samples,).
    solution_dts : numpy.ndarray or mlx.core.array
        1D array of time step sizes (dts) for each sample, with shape (num_samples,).
    batch_size : int
        The number of samples per batch.
    shuffle : bool, optional
        If True, the data will be shuffled before creating mini-batches. Defaults to True.

    Yields
    ------
    tuple
        A tuple containing:
        - src_tensor : mlx.core.array
            The source input tensor for the batch, representing the temperature distribution.
        - target_tensor : mlx.core.array
            The target output tensor for the batch (typically the same as the source for autoregressive training).
        - batch_alphas_tensor : mlx.core.array
            The thermal diffusivities (alphas) for the batch.
        - batch_dts_tensor : mlx.core.array
            The time step sizes (dts) for the batch.
    """
    num_samples = data.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)
    indices = mx.array(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        batch_data = data[batch_indices, :, :]
        batch_alphas = alphas[batch_indices]
        batch_dts = solution_dts[batch_indices]

        src_tensor = mx.array(batch_data)
        target_tensor = mx.array(batch_data)
        batch_alphas_tensor = mx.array(batch_alphas)
        batch_dts_tensor = mx.array(batch_dts)

        yield src_tensor, target_tensor, batch_alphas_tensor, batch_dts_tensor


def calculate_spatial_derivative_2D(T, dx, dy):
    """
    Calculates the second-order spatial derivatives of the temperature field in the x and y directions.

    This function computes the second-order central difference approximation of the spatial derivatives
    for the temperature distribution `T` over a 2D grid. The derivatives are calculated using finite
    differences with respect to the spatial steps `dx` (x-axis) and `dy` (y-axis).

    Parameters
    ----------
    T : numpy.ndarray or mlx.core.array
        The 4D array representing the temperature distribution, with shape (batch_size, time_steps, ny, nx), where:
        - batch_size is the number of samples,
        - time_steps is the number of time steps in the simulation,
        - ny is the number of grid points along the y-axis,
        - nx is the number of grid points along the x-axis.
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.

    Returns
    -------
    tuple of (numpy.ndarray or mlx.core.array, numpy.ndarray or mlx.core.array)
        A tuple containing:
        - d2T_dx2 : numpy.ndarray or mlx.core.array
            The second derivative of `T` with respect to the x direction, with shape (batch_size, time_steps, ny-2, nx-2).
        - d2T_dy2 : numpy.ndarray or mlx.core.array
            The second derivative of `T` with respect to the y direction, with shape (batch_size, time_steps, ny-2, nx-2).
    """
    d2T_dx2 = (T[:, :, 1:-1, 2:] - 2 * T[:, :, 1:-1, 1:-1] + T[:, :, 1:-1, :-2]) / dx ** 2
    d2T_dy2 = (T[:, :, 2:, 1:-1] - 2 * T[:, :, 1:-1, 1:-1] + T[:, :, :-2, 1:-1]) / dy ** 2
    return d2T_dx2, d2T_dy2


def calculate_temporal_derivative_2D(T, dt):
    """
    Calculates the temporal derivative of the temperature field over time.

    This function computes the first-order difference approximation of the temporal derivative
    for the temperature distribution `T` over a 2D grid. The derivative is calculated using
    finite differences with respect to the time steps `dt`. It assumes that the first dimension
    of `T` corresponds to time.

    Parameters
    ----------
    T : numpy.ndarray or mlx.core.array
        The 4D array representing the temperature distribution, with shape (batch_size, time_steps, ny, nx), where:
        - batch_size is the number of samples,
        - time_steps is the number of time steps in the simulation,
        - ny is the number of grid points along the y-axis,
        - nx is the number of grid points along the x-axis.
    dt : numpy.ndarray or mlx.core.array
        1D array of time step sizes for each sample, with shape (batch_size,). Each element represents
        the time step size for the corresponding sample.

    Returns
    -------
    numpy.ndarray or mlx.core.array
        The temporal derivative of the temperature field `T` with respect to time, with shape
        (batch_size, time_steps - 1, ny, nx).
    """
    dt_reshaped = dt.reshape(-1, 1, 1, 1)
    dT_dt = (T[:, 1:, :, :] - T[:, :-1, :, :]) / dt_reshaped
    return dT_dt


def physics_informed_loss_2D(model_output, src_alphas, src_dts, dx, dy):
    """
    Computes the physics-informed loss for a 2D heat diffusion model.

    This function calculates the physics-informed loss by enforcing the heat equation on the model's
    output. It aligns the model output to the grid, computes the second-order spatial derivatives
    and the first-order temporal derivative, and then calculates the residuals between the temporal
    derivative and the spatial derivatives scaled by the thermal diffusivity. The loss is the mean
    squared error of these residuals.

    Parameters
    ----------
    model_output : numpy.ndarray or mlx.core.array
        The output from the model, representing the predicted temperature distribution field.
        It has the shape (batch_size, time_steps, ny, nx).
    src_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values (alphas) for each sample, with shape (batch_size,).
    src_dts : numpy.ndarray or mlx.core.array
        The time step sizes (dts) for each sample, with shape (batch_size,).
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.

    Returns
    -------
    mx.core.array
        The physics-informed loss value, calculated as the mean squared error of the residuals between
        the temporal derivative and the scaled spatial derivatives.
    """
    model_output_pre_aligned = model_output[:, :, 1:-1, 1:-1]

    d2T_dx2, d2T_dy2 = calculate_spatial_derivative_2D(model_output, dx, dy)
    dT_dt = calculate_temporal_derivative_2D(model_output_pre_aligned, src_dts)

    alphas_reshaped = src_alphas.reshape(-1, 1, 1, 1)
    residuals = dT_dt - alphas_reshaped * (d2T_dx2[:, :-1, :, :] + d2T_dy2[:, :-1, :, :])

    residual_std = mx.sqrt(mx.var(residuals) + 1e-8)
    normalized_residuals = residuals  # / residual_std

    pi_loss = nn.losses.mse_loss(normalized_residuals, mx.zeros_like(normalized_residuals), reduction='mean')

    return pi_loss


def compute_boundary_loss_2D(model_output, target):
    """
    Computes the boundary loss for a 2D heat diffusion model.

    This function calculates the mean squared error (MSE) between the predicted boundary conditions
    (left, right, top, and bottom) and the expected boundary conditions (from the target). The boundary
    conditions are compared for each time step and sample in the batch. The total boundary loss is the sum
    of the individual MSE losses for the four boundaries.

    Parameters
    ----------
    model_output : numpy.ndarray or mlx.core.array
        The predicted temperature distribution field from the model, with shape
        (batch_size, time_steps, ny, nx). The last two dimensions represent the spatial grid.
    target : numpy.ndarray or mlx.core.array
        The expected temperature distribution field, with the same shape as `model_output`.

    Returns
    -------
    mx.core.array
        The boundary loss, computed as the sum of the MSE losses for the left, right, top, and bottom boundaries.
    """
    expected_left_boundary = target[:, :, :, 0]
    expected_right_boundary = target[:, :, :, -1]
    expected_top_boundary = target[:, :, 0, :]
    expected_bottom_boundary = target[:, :, -1, :]

    left_boundary_pred = model_output[:, :, :, 0]
    right_boundary_pred = model_output[:, :, :, -1]
    top_boundary_pred = model_output[:, :, 0, :]
    bottom_boundary_pred = model_output[:, :, -1, :]

    left_boundary_loss = nn.losses.mse_loss(left_boundary_pred, expected_left_boundary,
                                            reduction='mean')
    right_boundary_loss = nn.losses.mse_loss(right_boundary_pred, expected_right_boundary,
                                             reduction='mean')
    top_boundary_loss = nn.losses.mse_loss(top_boundary_pred, expected_top_boundary,
                                           reduction='mean')
    bottom_boundary_loss = nn.losses.mse_loss(bottom_boundary_pred, expected_bottom_boundary,
                                              reduction='mean')

    boundary_loss = left_boundary_loss + right_boundary_loss + top_boundary_loss + bottom_boundary_loss
    return boundary_loss


def compute_initial_loss_2D(model_output, target):
    """
    Computes the initial frame loss for a 2D heat diffusion model.

    This function calculates the mean squared error (MSE) between the predicted and expected initial
    frames (the first few time steps) of the temperature distribution field. The initial frames are
    compared for each sample in the batch.

    Parameters
    ----------
    model_output : numpy.ndarray or mlx.core.array
        The predicted temperature distribution field from the model, with shape
        (batch_size, time_steps, ny, nx). The last two dimensions represent the spatial grid.
    target : numpy.ndarray or mlx.core.array
        The expected temperature distribution field, with the same shape as `model_output`.

    Returns
    -------
    mx.core.array
        The initial frame loss, computed as the mean squared error between the predicted and expected initial frames.
    """
    expected_initial_frames = target[:, 0:5, :, :]
    initial_frames_predicted = model_output[:, 0:5, :, :]

    initial_frames_loss = nn.losses.mse_loss(initial_frames_predicted,
                                             expected_initial_frames, reduction='mean')

    return initial_frames_loss


def loss_fn_2D(model, src, target, src_alphas, src_dts, dx, dy):
    """
    Calculates the total loss for a 2D heat diffusion model.

    This function computes the total loss as a weighted sum of several loss components:
    - Mean Squared Error (MSE) between the predicted and target temperature fields.
    - Physics-informed loss, which enforces the heat equation on the model output.
    - Boundary condition loss, which measures the deviation from expected boundary conditions.
    - Initial condition loss, which compares the first few time steps of the predicted and target fields.

    The total loss is weighted by predefined factors for each component (boundary, physics, and initial losses).

    Parameters
    ----------
    model : object
        The 2D heat diffusion model that predicts the temperature distribution.
    src : numpy.ndarray or mlx.core.array
        The input data representing the initial temperature distribution, with shape
        (batch_size, time_steps, ny, nx).
    target : numpy.ndarray or mlx.core.array
        The expected output temperature distribution, with the same shape as `src`.
    src_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample, with shape (batch_size,).
    src_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample, with shape (batch_size,).
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.

    Returns
    -------
    mx.core.array
        The total loss, which is a weighted sum of the MSE loss, physics-informed loss, boundary condition loss,
        and initial condition loss.
    """
    boundary_loss_weight = 0.1
    physics_loss_weight = 0.001
    initial_loss_weight = 0.1

    model_output = model(src, src_alphas)

    mse_loss = nn.losses.mse_loss(model_output, target, reduction='mean')
    pi_loss = physics_informed_loss_2D(model_output, src_alphas, src_dts, dx, dy)
    boundary_loss = compute_boundary_loss_2D(model_output, target)
    initial_loss = compute_initial_loss_2D(model_output, target)

    total_loss = (mse_loss + boundary_loss_weight * boundary_loss
                  + physics_loss_weight * pi_loss + initial_loss_weight * initial_loss)

    return total_loss


def get_learning_rate_for_epoch(epoch, schedule):
    """
    Function get_learning_rate_for_epoch.
    Args:
        epoch: Description of epoch.
        schedule: Description of schedule.
    Returns:
        Loaded data or configuration.
    """
    sorted_epochs = sorted(schedule.keys())
    for i in range(len(sorted_epochs) - 1):
        if sorted_epochs[i] <= epoch - 1 < sorted_epochs[i + 1]:
            return schedule[sorted_epochs[i]]
    return schedule[sorted_epochs[-1]] if epoch >= sorted_epochs[-1] else 0.0


# Training and validation function with periodic saving
def train_and_validate(train_data, train_alphas, train_dts, validation_data, validation_alphas, validation_dts,
                       batch_size, epochs, start_epoch, save_interval, dx, dy):
    """
    Trains and validates a 2D heat diffusion model over a specified number of epochs.

    This function performs the training and validation of a 2D heat diffusion model. It iterates over
    the training and validation datasets in mini-batches, calculates the loss, and updates the model
    parameters using the optimizer. The model and optimizer states are periodically saved, and
    the training and validation losses are printed after each epoch. If training is resumed from
    a checkpoint, the model and optimizer are reloaded, and the learning rate is adjusted accordingly.

    Parameters
    ----------
    train_data : numpy.ndarray or mlx.core.array
        The training dataset representing the temperature distribution over time. The shape is
        (num_samples, time_steps, ny, nx).
    train_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample in the training dataset, with shape (num_samples,).
    train_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample in the training dataset, with shape (num_samples,).
    validation_data : numpy.ndarray or mlx.core.array
        The validation dataset representing the temperature distribution over time. The shape is
        (num_samples, time_steps, ny, nx).
    validation_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample in the validation dataset, with shape (num_samples,).
    validation_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample in the validation dataset, with shape (num_samples,).
    batch_size : int
        The number of samples per batch during training and validation.
    epochs : int
        The total number of epochs to train the model.
    start_epoch : int
        The epoch from which to resume training. Typically 0 for a fresh start or the epoch from
        which training is resumed after a checkpoint.
    save_interval : int
        The number of epochs between saving the model and optimizer states.
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.

    Returns
    -------
    None
        The function performs training and validation and does not return any value. It prints
        training and validation losses and saves the model and optimizer states at regular intervals.
    """
    # if start_epoch > 1:   # if this is a reload run, do two warm-up iterations first
    #    for epoch in range(start_epoch-2, start_epoch):
    #        optimizer.learning_rate = 0
    #        for src, target, src_alphas, src_dts in data_loader_2D(train_data, train_alphas, train_dts, batch_size,
    #                                                               shuffle=False):
    #            _, _ = train_step(src, target, src_alphas, src_dts, dx, dy)
    #            mx.eval(state)

    # Set learning rate to match the reload epoch (caution: filenames are based on epoch+1)
    # optimizer.learning_rate = get_learning_rate_for_epoch(start_epoch, learning_rate_schedule)

    tic = time.perf_counter()
    print(f'start_epoch: {start_epoch} epochs: {epochs} learning_rate: {optimizer.learning_rate}')

    for epoch in range(start_epoch, epochs):
        if epoch in config["learning_rate_schedule"]:
            optimizer.learning_rate = config["learning_rate_schedule"][epoch]

        total_train_loss = 0
        num_train_batches = 0

        for src, target, src_alphas, src_dts in data_loader_2D(train_data, train_alphas, train_dts, batch_size,
                                                               shuffle=True):
            loss, grads = train_step(src, target, src_alphas, src_dts, dx, dy)
            total_train_loss += loss.item()
            num_train_batches += 1
            mx.eval(state)

        total_val_loss = 0
        num_val_batches = 0
        model.eval()

        for src, target, src_alphas, src_dts in data_loader_2D(validation_data, validation_alphas, validation_dts,
                                                               batch_size, shuffle=False):
            val_loss = evaluate_step(src, target, src_alphas, src_dts, dx, dy)
            total_val_loss += val_loss.item()
            num_val_batches += 1

        print(
            f'Epoch {epoch + 1}, lr: {optimizer.learning_rate}, '
            f'Training Loss: {total_train_loss / num_train_batches}, '
            f'Validation Loss: {total_val_loss / num_val_batches}, '
            f'Number of Train batches: {num_train_batches}')

        # Save the model and configuration periodically and live reload to check consistency
        if (epoch + 1) % save_interval == 0:
            mx.eval(model.parameters(), optimizer.state)
            mx.eval(state)
            model.eval()
            save_model_and_optimizer(config, epoch + 1, save_dir_path, model_base_file_name,
                                     optimizer_base_file_name, hyper_base_file_name)
        model.train()
    toc = time.perf_counter()
    tpi = (toc - tic) / 60 / (epochs + 1 - start_epoch)
    print(f"Time per epoch {tpi:.3f} (min)")


def evaluate_model(model, data_loader_func, test_data, test_alphas, test_dts, dx, dy, batch_size):
    """
    Evaluates the performance of the model on the test dataset.

    This function evaluates the model by calculating the loss on the test dataset. It uses the
    provided `data_loader_func` to load the test data in mini-batches and computes the average
    loss across all batches.

    Parameters
    ----------
    model : object
        The trained 2D heat diffusion model that will be evaluated.
    data_loader_func : function
        The function that generates mini-batches from the dataset. It should accept the test data,
        alphas, time steps, batch size, and shuffle option.
    test_data : numpy.ndarray or mlx.core.array
        The test dataset representing the temperature distribution over time. The shape is
        (num_samples, time_steps, ny, nx).
    test_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample in the test dataset, with shape (num_samples,).
    test_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample in the test dataset, with shape (num_samples,).
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.
    batch_size : int
        The number of samples per batch during evaluation.

    Returns
    -------
    float
        The average loss over the entire test dataset. If no batches are present, returns `float('inf')`.
    """
    total_loss = 0
    num_batches = 0

    data_loader = data_loader_func(test_data, test_alphas, test_dts, batch_size, shuffle=False)

    for src, target, src_alphas, src_dts in data_loader:
        loss = evaluate_step(src, target, src_alphas, src_dts, dx, dy)
        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Average Test Loss: {average_loss}")
    return average_loss


def evaluate_self_regressive_model_BeyondL(model, data_loader_func, test_data, test_alphas, test_dts, dx, dy,
                                           batch_size, seq_len, output_dir_regress="./MSE_step_regress"):
    """
    Evaluates the model in a self-regressive manner and tracks MSE for each predicted time step.

    This function evaluates a 2D heat diffusion model by performing self-regressive prediction,
    tracking the Mean Squared Error (MSE) for each time step. In self-regressive evaluation, the model
    predicts a sequence of temperature distributions, where each predicted frame is used as input to predict
    future frames. The function accumulates the MSE loss for each time step across all batches, and
    produces a plot of MSE evolution over time steps.

    Parameters
    ----------
    model : object
        The trained 2D heat diffusion model to be evaluated.
    data_loader_func : function
        The function that generates mini-batches from the dataset. It should accept the test data,
        alphas, time steps, batch size, and shuffle option.
    test_data : numpy.ndarray or mlx.core.array
        The test dataset representing the temperature distribution over time. The shape is
        (num_samples, time_steps, ny, nx).
    test_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample in the test dataset, with shape (num_samples,).
    test_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample in the test dataset, with shape (num_samples,).
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.
    batch_size : int
        The number of samples per batch during evaluation.
    seq_len : int
        The total number of time steps (sequence length) for the evaluation.
    output_dir_regress : str, optional
        The directory where the MSE plots will be saved. Defaults to "./MSE_step_regress".

    Returns
    -------
    None

    This function tracks the MSE for each time step and generates a plot showing the evolution
    of MSE across time steps. The plot is saved as an image in the specified output directory.
    """
    n_replace = 5
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(output_dir_regress):
        os.makedirs(output_dir_regress)

    data_loader = data_loader_func(test_data, test_alphas, test_dts, batch_size, shuffle=False)

    # Initialize MSE tracker for all time steps
    cumulative_mse = np.zeros(seq_len - 5)
    num_batches = 0

    for src, target, src_alphas, src_dts in data_loader:
        # Initialize per-batch MSE tracker for all time steps
        time_step_mse = np.zeros(seq_len - 5)

        # Loop through time steps autoregressively, replacing predictions
        for t in range(5, seq_len, n_replace):
            prediction = model(src, src_alphas)
            end_idx = min(t + n_replace, seq_len)
            src[:, t:end_idx, :, :] = prediction[:, t:end_idx, :, :]

        # Track and accumulate MSE for each time step (starting from t=5)
        for t in range(5, seq_len):
            mse_loss = nn.losses.mse_loss(prediction[:, t, :, :], target[:, t, :, :], reduction="mean")
            time_step_mse[t - 5] = mse_loss.item()  # Store MSE for the current batch
            cumulative_mse[t - 5] += mse_loss.item()  # Accumulate MSE for this time step

        num_batches += 1
        print(f'finished batch: {num_batches}')

    # After looping through all batches:
    # Average the MSE by the number of batches
    average_mse = cumulative_mse / num_batches
    print(f"Average MSE over auto-regressive sequence {average_mse}")

    # Plot MSE evolution over time steps
    plt.figure(figsize=(10, 6))
    plt.plot(range(5, seq_len), average_mse, marker='o', linestyle='-', color='b')
    plt.xlabel("Time Step")
    plt.ylabel("MSE Loss")
    plt.title("Average MSE Evolution Over Autoregressive Time Steps")
    plt.grid(True)
    mse_plot_filename = os.path.join(output_dir_regress, "mse_evolution.png")
    plt.savefig(mse_plot_filename)
    plt.close()

    print(f"MSE for each time step saved in {mse_plot_filename}")



def evaluate_model_block_sequence(model, data_loader_func, test_data, test_alphas, test_dts, dx, dy,
                                 batch_size, seq_len, output_dir_block="./MSE_step_block"):
    """
    Evaluates the model by predicting the entire sequence at once and tracks MSE for each predicted time step.
    This version assumes the model predicts the entire sequence at once (not autoregressively).
    """
    batch_size = 4
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(output_dir_block):
        os.makedirs(output_dir_block)

    data_loader = data_loader_func(test_data, test_alphas, test_dts, batch_size, shuffle=False)

    # Initialize MSE tracker for all time steps
    cumulative_mse = np.zeros(seq_len - 5)
    num_batches = 0

    for src, target, src_alphas, src_dts in data_loader:
        # Initialize per-batch MSE tracker for all time steps
        time_step_mse = np.zeros(seq_len - 5)

        # Get full sequence predictions
        prediction = model(src, src_alphas)

        # Track and accumulate MSE for each time step (starting from t=5)
        for t in range(5, seq_len):
            mse_loss = nn.losses.mse_loss(prediction[:, t, :, :], target[:, t, :, :], reduction="mean")
            time_step_mse[t - 5] = mse_loss.item()  # Store MSE for the current batch
            cumulative_mse[t - 5] += mse_loss.item()  # Accumulate MSE for this time step

        num_batches += 1

    # After looping through all batches:
    # Average the MSE by the number of batches
    average_mse = cumulative_mse / num_batches
    print(f"Average MSE over block sequence {average_mse}")

    # Plot MSE evolution over time steps
    plt.figure(figsize=(10, 6))
    plt.plot(range(5, seq_len), average_mse, marker='o', linestyle='-', color='b')
    plt.xlabel("Time Step")
    plt.ylabel("MSE Loss")
    plt.title("Average MSE Evolution Over Full Sequence Prediction")
    plt.grid(True)
    mse_plot_filename = os.path.join(output_dir_block, "mse_evolution_full_sequence.png")
    plt.savefig(mse_plot_filename)
    plt.close()

    print(f"MSE for each time step saved in {mse_plot_filename}")

def plot_predictions_2D(model, data_loader_func, data_loader_args, num_examples=5, output_dir="./frames2D",
                        t_trained=1200):
    """
    Generates and saves model predictions vs. actual data for qualitative evaluation.

    This function plots and saves the model's predictions against the actual temperature distribution data
    for a specified number of examples. For each time step, the actual temperature and the predicted
    temperature are plotted side by side, allowing for a qualitative comparison. The generated images
    are saved in the specified output directory.

    Parameters
    ----------
    model : object
        The trained 2D heat diffusion model to generate predictions.
    data_loader_func : function
        The function that loads the test data in mini-batches. It should accept the necessary
        arguments through `data_loader_args`.
    data_loader_args : dict
        A dictionary containing the arguments to be passed to the `data_loader_func`, including the
        test data, alphas, time steps, batch size, and shuffle option.
    ny : int
        The number of grid points along the y-axis (height of the 2D grid).
    nx : int
        The number of grid points along the x-axis (width of the 2D grid).
    num_examples : int, optional
        The number of examples to plot. Defaults to 5.
    output_dir : str, optional
        The directory where the generated images will be saved. Defaults to "./frames2D".
    t_trained : int, optional
        The time step until which the model was trained. Time steps beyond this value will be
        marked as extrapolated in the plot title. Defaults to 1200.

    Returns
    -------
    None
        The function generates and saves comparison plots of actual vs. predicted temperature distributions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(output_dir)

    for i, (src, target, test_alphas, test_dts) in enumerate(data_loader_func(**data_loader_args)):
        if i >= num_examples:
            break

        prediction = model(src, test_alphas)  # .reshape(-1, src.shape[1], ny, nx)

        for t in range(target.shape[1]):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            actual_temp = src[0, t, :, :]
            im0 = axs[0].imshow(actual_temp, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
            axs[0].set_title(f'Actual Temp, Time Step {t}')
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            predicted_temp = prediction[0, t, :, :]
            im1 = axs[1].imshow(predicted_temp, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
            axs[1].set_title(f'Predicted Temp, Time Step {t}')
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            plt.suptitle(f'Example {i + 1}: alpha {test_alphas[0].item():.5f}' + (
                ', Extrapolated' if t_trained is not None and t > t_trained else ''))
            plt.tight_layout()

            frame_filename = os.path.join(output_dir, f"example_{i + 1}_step_{t}.png")
            plt.savefig(frame_filename)
            # plt.show()

            plt.close(fig)
        print(f'Created heatmaps for Test Example: {i+1}')

def plot_regressive_predictions_2D(model, data_loader_func, data_loader_args, num_examples=5,
                                   output_dir="./frames2D_regress", t_trained=1200):
    """
    Evaluate the model in a self-regressive manner and plot predictions.
    """
    total_loss = 0
    num_batches = 0
    num_plots = 0
    t_trained = None
    n_replace = 5
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #data_loader = data_loader_func(test_data, test_alphas, test_dts, batch_size, shuffle=False)

    for i, (src, target, test_alphas, test_dts) in enumerate(data_loader_func(**data_loader_args)):
        if i >= num_examples:
            break
        for t in range(5, target.shape[1], n_replace):
            prediction = model(src, test_alphas)
            end_idx = min(t + n_replace, target.shape[1])
            src[:, t:end_idx, :, :] = prediction[:, t:end_idx, :, :]

        num_batches += 1

        for t in range(target.shape[1]):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            actual_temp = target[0, t, :, :]
            im0 = axs[0].imshow(actual_temp, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
            axs[0].set_title(f'Actual Temp, Time Step {t + 1}')
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            predicted_temp = prediction[0, t, :, :]
            im1 = axs[1].imshow(predicted_temp, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
            axs[1].set_title(f'Predicted Temp, Time Step {t + 1}')
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            plt.suptitle(f'Example {num_plots + 1}: alpha {test_alphas[0].item():.5f}'
                         + (', !Forward Extrapolation!' if t_trained is not None and t > t_trained - 1 else ''))
            plt.tight_layout()

            frame_filename = os.path.join(output_dir, f"example_{num_plots + 1}_step_{t + 1}.png")
            plt.savefig(frame_filename)
            plt.close(fig)
        num_plots += 1
        print(f'Created heatmaps for Test Example: {num_plots}')


def plot_positional_encodings(model, seq_len):
    """
    Plots the positional encodings for the spatial (y, x) and temporal dimensions from the model.

    This function visualizes the positional encodings learned by the model for the y-axis, x-axis,
    and time dimension (temporal). The positional encodings are summed across the embedding dimension
    and plotted as heatmaps to show how the encodings change across the respective dimensions.

    Parameters
    ----------
    model : object
        The trained 2D heat diffusion model, which contains positional encodings for the spatial
        (y and x axes) and temporal (time steps) dimensions.
    seq_len : int
        The total number of time steps (sequence length) for the temporal positional encoding.

    Returns
    -------
    None
        The function generates and displays plots of the positional encodings for the y-axis, x-axis,
        and temporal dimensions.
    """
    ny, nx = model.ny, model.nx
    # Generate positional indices for the full range
    pos_indices_y = mx.arange(ny).reshape(1, -1)  # Shape: (1, ny)
    pos_indices_x = mx.arange(nx).reshape(-1, 1)  # Shape: (nx, 1)
    pos_indices_t = mx.arange(seq_len).reshape(1, -1)  # Shape: (1, seq_len)

    # Get positional encodings from the model
    pos_enc_y = np.array(model.positional_encoding_y(pos_indices_y))  # Should be (1, ny, embed_dim)
    pos_enc_x = np.array(model.positional_encoding_x(pos_indices_x))  # Should be (nx, 1, embed_dim)
    pos_enc_t = np.array(model.positional_encoding_t(pos_indices_t))  # Should be (1, seq_len, embed_dim)

    # Sum across embedding dimension to reduce to 2D
    pos_enc_y_sum = pos_enc_y.sum(axis=2)  # Sum over embedding dimension
    pos_enc_x_sum = pos_enc_x.sum(axis=2)
    pos_enc_t_sum = pos_enc_t.sum(axis=2)  # Sum over embedding dimension

    import matplotlib.pyplot as plt

    plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.title('Positional Encoding Y')
    plt.imshow(pos_enc_y_sum, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title('Positional Encoding X')
    plt.imshow(pos_enc_x_sum.T, aspect='auto', cmap='viridis')  # Transposed to align dimensions
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title('Temporal Positional Encoding')
    plt.imshow(pos_enc_t_sum, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.show()


def compare_dict_states(original, loaded, state_name):
    """
    Compares two dictionary states and logs differences.

    This function compares the values in two dictionaries (`original` and `loaded`) to check for
    mismatches. It supports nested dictionaries and lists within the dictionaries. The function
    also handles cases where keys or values are `None`, empty lists, or empty dictionaries, and
    skips logging errors for such cases. The function returns `True` if all keys and values match
    between the two dictionaries, and `False` otherwise.

    Parameters
    ----------
    original : dict
        The original dictionary state to be compared.
    loaded : dict
        The loaded dictionary state to be compared against the original.
    state_name : str
        A string representing the name of the state being compared, used for logging purposes.

    Returns
    -------
    bool
        `True` if the two dictionaries match in all keys and values, `False` otherwise.
    """
    match = True
    for key in original:
        if key not in loaded:
            if original[key] is None or original[key] == [] or original[key] == {}:
                continue  # Do not treat as an error if the original value is None or empty
            print(f"Error comparing {state_name} at key: {key} - Key not found in loaded state.")
            match = False
            continue
        if isinstance(original[key], dict) and isinstance(loaded[key], dict):
            if not compare_dict_states(original[key], loaded[key], f"{state_name}.{key}"):
                match = False
        elif isinstance(original[key], list) and isinstance(loaded[key], list):
            if not compare_list_states(original[key], loaded[key], f"{state_name}.{key}"):
                match = False
        else:
            if original[key] is None and loaded[key] is None:
                continue
            if not np.array_equal(original[key], loaded[key]):
                print(f"Error comparing {state_name} at key: {key}")
                match = False
    for key in loaded:
        if key not in original:
            print(f"Error comparing {state_name} at key: {key} - Key not found in original state.")
            match = False
    return match


def compare_list_states(original, loaded, state_name):
    """
    Compares two list states and logs differences.

    This function compares two lists (`original` and `loaded`) element by element, logging any
    mismatches. It supports lists that may contain dictionaries or nested lists. The function
    also handles cases where elements are `None` and skips logging errors for such cases. If
    the lists have different lengths, it logs an error and returns `False`. The function returns
    `True` if the lists match in both length and content, and `False` otherwise.

    Parameters
    ----------
    original : list
        The original list state to be compared.
    loaded : list
        The loaded list state to be compared against the original.
    state_name : str
        A string representing the name of the state being compared, used for logging purposes.

    Returns
    -------
    bool
        `True` if the two lists match in length and content, `False` otherwise.
    """
    match = True
    if len(original) != len(loaded):
        print(f"Error comparing {state_name} - Length mismatch.")
        return False
    for i in range(len(original)):
        if isinstance(original[i], dict) and isinstance(loaded[i], dict):
            if not compare_dict_states(original[i], loaded[i], f"{state_name}[{i}]"):
                match = False
        elif isinstance(original[i], list) and isinstance(loaded[i], list):
            if not compare_list_states(original[i], loaded[i], f"{state_name}[{i}]"):
                match = False
        else:
            if original[i] is None and loaded[i] is None:
                continue
            if not np.array_equal(original[i], loaded[i]):
                print(f"Error comparing {state_name} at index: {i}")
                match = False
    return match


# Function to save model and optimizer state periodically
def save_model_and_optimizer(config, current_epoch, dir_path, model_base_file_name, optimizer_base_file_name,
                             hyper_base_file_name):
    """
    Saves the model's state, weights, optimizer state, random state, and configuration at the specified epoch.

    This function saves the current state of the model, including its weights, the optimizer's state,
    the random state, and the training configuration to files. These files can be used to resume
    training or perform model evaluation at a later point.

    Parameters
    ----------
    config : dict
        The training configuration, including parameters like batch size, learning rate, and model architecture.
    current_epoch : int
        The current epoch number, used to save the model and optimizer states with the corresponding epoch suffix.
    dir_path : str
        The directory where the model, optimizer, random state, and configuration files will be saved.
    model_base_file_name : str
        The base name for the model file (the epoch number will be appended).
    optimizer_base_file_name : str
        The base name for the optimizer file (the epoch number will be appended).
    hyper_base_file_name : str
        The base name for the configuration (hyperparameters) file (the epoch number will be appended).

    Returns
    -------
    None
        The function saves the model's state, optimizer state, random state, and configuration to files in the specified directory.
    """
    # Ensure the directory exists, create it if not
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    model_file_name = f"{model_base_file_name}_epoch_{current_epoch}.pkl"
    weights_file_name = f"{model_base_file_name}_weights_epoch_{current_epoch}.safetensors"
    # optimizer_file_name = f"{optimizer_base_file_name}_epoch_{current_epoch}.safetensors"
    optimizer_file_name = f"{optimizer_base_file_name}_epoch_{current_epoch}.pkl"
    random_state_file_name = f"random_state_epoch_{current_epoch}.pkl"
    config_file_name = f"{hyper_base_file_name}_epoch_{current_epoch}.json"

    # Save model state (parameters only)
    model_file_path = os.path.join(dir_path, model_file_name)
    with open(model_file_path, 'wb') as f:
        pickle.dump(model.parameters(), f)
        # pickle.dump(model.state, f)

    # Save model weights
    weights_file_path = os.path.join(dir_path, weights_file_name)
    model.save_weights(weights_file_path)

    # Save optimizer state
    optimizer_file_path = os.path.join(dir_path, optimizer_file_name)
    with open(optimizer_file_path, 'wb') as f:
        pickle.dump(optimizer.state, f)

    # Save random state
    random_state_path = os.path.join(dir_path, random_state_file_name)
    with open(random_state_path, 'wb') as f:
        pickle.dump(mx.random.state, f)

    # Save training configuration
    config['current_epoch'] = current_epoch
    hyper_file_path = os.path.join(dir_path, config_file_name)
    with open(hyper_file_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)

    print(f"Model, optimizer, random state, and configuration saved at epoch {current_epoch}.")


def load_model_and_optimizer(model, optimizer, dir_path, model_base_file_name, optimizer_base_file_name,
                             hyper_base_file_name, checkpoint_epoch, comparison=config["compare_current_loaded"]):
    """
    Loads the model state, optimizer state, random state, and configuration from a specific checkpoint.

    This function loads the saved model parameters, optimizer state, random state, and training configuration
    from a specified epoch. It checks for the existence of the saved files and compares the current states with
    the loaded ones to ensure consistency. If no checkpoint is found, the training starts from scratch.

    Parameters
    ----------
    model : object
        The model instance whose state (parameters and weights) will be loaded.
    optimizer : object
        The optimizer instance whose state will be loaded.
    dir_path : str
        The directory where the model, optimizer, random state, and configuration files are saved.
    model_base_file_name : str
        The base file name used for saving the model (with epoch appended).
    optimizer_base_file_name : str
        The base file name used for saving the optimizer (with epoch appended).
    hyper_base_file_name : str
        The base file name used for saving the configuration (with epoch appended).
    checkpoint_epoch : int
        The epoch number from which to load the model, optimizer, random state, and configuration.

    Returns
    -------
    tuple
        A tuple containing:
        - start_epoch : int
            The epoch from which to resume training. If no checkpoint is found, this is set to 0.
        - loaded_optimizer_state : dict
            The loaded optimizer state.
        - loaded_random_state : object
            The loaded random state for reproducibility.
        - loaded_parameters : dict
            The loaded model parameters.
        - loaded_config: dict
            The loaded configration
    """
    # Construct the file paths for model, optimizer, random state, and configuration
    model_file_name = f"{model_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    weights_file_name = f"{model_base_file_name}_weights_epoch_{checkpoint_epoch}.safetensors"
    optimizer_file_name = f"{optimizer_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    random_state_file_name = f"random_state_epoch_{checkpoint_epoch}.pkl"
    config_file_name = f"{hyper_base_file_name}_epoch_{checkpoint_epoch}.json"

    model_file_path = os.path.join(dir_path, model_file_name)
    weights_file_path = os.path.join(dir_path, weights_file_name)
    optimizer_file_path = os.path.join(dir_path, optimizer_file_name)
    random_state_file_path = os.path.join(dir_path, random_state_file_name)
    config_file_path = os.path.join(dir_path, config_file_name)

    # Check if all necessary files exist for loading
    if os.path.exists(model_file_path) and os.path.exists(optimizer_file_path) and os.path.exists(
            random_state_file_path) and os.path.exists(config_file_path) and os.path.exists(weights_file_path):
        # Load model state (parameters only)
        with open(model_file_path, 'rb') as f:
            loaded_parameters = pickle.load(f)

        # Load optimizer state
        with open(optimizer_file_path, 'rb') as f:
            loaded_optimizer_state = pickle.load(f)

        # Load random state
        with open(random_state_file_path, 'rb') as f:
            loaded_random_state = pickle.load(f)

        # Load training configuration
        with open(config_file_path, 'r') as json_file:
            loaded_config = json.load(json_file)
            #config = loaded_config

        # Get the start epoch from the configuration
        start_epoch = loaded_config.get('current_epoch', 0)

        # Load current states to compare with the loaded states
        current_optimizer_state = optimizer.state
        current_random_state = mx.random.state
        current_model_parameters = model.parameters()

        # Compare the states for consistency (useful when live saving-and_reloading during training for debugging)
        if comparison:
            if compare_dict_states(current_optimizer_state, loaded_optimizer_state, 'optimizer state'):
                print('Optimizer state matches.')
            else:
                print("Optimizer state mismatch detected.")

            if compare_list_states(current_random_state, loaded_random_state, 'random state'):
                print("Random state matches.")
            else:
                print("Random state mismatch detected.")

            if compare_dict_states(current_model_parameters, loaded_parameters, 'model state'):
                print("Model state matches.")
            else:
                print("Model state mismatch detected.")

            print(f"Model, optimizer, random state, and configuration loaded from {dir_path} at epoch {checkpoint_epoch}.")
    else:
        # If no checkpoint is found, start from scratch
        start_epoch = 0
        print(f"No saved model found at epoch {checkpoint_epoch}. Starting training from scratch.")

    return start_epoch, loaded_optimizer_state, loaded_random_state, loaded_parameters, loaded_config


def plot_model_weights(model, epoch):
    """
    Plots statistics of the model's weights and saves the plots.

    This function extracts, processes, and visualizes the model's weight statistics. It filters the model's weights,
    applies transformations (e.g., mean, max, count of weights above a threshold), and plots the results as heatmaps.
    The plots are saved as PNG files with the epoch number included in the filename.

    Parameters
    ----------
    model : object
        The trained model whose weights will be extracted and visualized. The model should contain an
        'output_projection' layer from which the weights are visualized.
    epoch : int
        The epoch number used to label the saved plots.

    Returns
    -------
    None
        The function generates and saves heatmaps of the weight statistics (mean, max, and counts above thresholds)
        but does not return any value.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set the model to evaluation mode
    model.eval()

    # Helper function to check if a value is an MLX array (adjust according to actual MLX type)
    def is_mlx_array(value):
        # Replace 'mx.array' with the actual array type in the MLX framework
        return isinstance(value, mx.array)

    # Function to filter the model's weights based on a condition
    def filter_fn(module, key, value):
        # Check if the key contains 'weight' and is a valid MLX array
        if isinstance(value, dict) and 'weight' in value:
            weight = value['weight']
            if is_mlx_array(weight):
                mean_abs = weight.abs().mean()
                return mean_abs > 0.003  # Filter out small weights
        elif is_mlx_array(value):
            mean_abs = value.abs().mean()
            return mean_abs > 0.003
        return False

    # Function to modify the weights during filtering
    def map_fn(value):
        if is_mlx_array(value):
            return value * 10  # Example scaling
        return value

    # Filter and modify the model's weights
    filtered_weights = model.filter_and_map(filter_fn, map_fn)

    # Retrieve weights from the 'output_projection' layer
    weights_dict = filtered_weights.get('output_projection', None)
    if weights_dict is not None:
        weights = weights_dict.get('weight', None)
        if weights is not None:
            # Reshape weights if needed
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)

            # Calculate absolute values of weights
            weights_abs = mx.abs(weights)
            weights_abs_reshaped = weights_abs.reshape(26, 26, 512)

            # Calculate min, mean, and max across the embedding dimension
            min_data = mx.min(weights_abs_reshaped, axis=2)
            mean_data = mx.mean(weights_abs_reshaped, axis=2)
            max_data = mx.max(weights_abs_reshaped, axis=2)

            # Threshold for significant weights
            threshold = 0.75 * max_data
            count_above_threshold = mx.sum(weights_abs_reshaped > threshold[:, :, None], axis=2)
            count_above_mean = mx.sum(weights_abs_reshaped > mean_data[:, :, None], axis=2)

            # Plot the results as heatmaps
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
            xticks = np.arange(0, 26, 2)
            yticks = np.arange(0, 26, 2)

            # Heatmap for the mean of absolute weights
            sns.heatmap(np.array(mean_data), ax=axes[0, 0], cmap='viridis', square=True,
                        cbar_ax=fig.add_axes([0.48, 0.53, 0.02, 0.35]))
            axes[0, 0].set_title('Mean of Abs Weights')
            axes[0, 0].set_xticks(xticks)
            axes[0, 0].set_yticks(yticks)

            # Heatmap for the max of absolute weights
            sns.heatmap(np.array(max_data), ax=axes[0, 1], cmap='viridis', square=True,
                        cbar_ax=fig.add_axes([0.903, 0.53, 0.02, 0.35]))
            axes[0, 1].set_title('Max of Abs Weights')
            axes[0, 1].set_xticks(xticks)
            axes[0, 1].set_yticks(yticks)

            # Heatmap for count above the mean
            sns.heatmap(np.array(count_above_mean), ax=axes[1, 0], cmap='viridis', square=True,
                        cbar_ax=fig.add_axes([0.48, 0.11, 0.015, 0.35]))
            axes[1, 0].set_title('Count Above Mean Value')
            axes[1, 0].set_xticks(xticks)
            axes[1, 0].set_yticks(yticks)

            # Heatmap for count above 0.75 * max value
            sns.heatmap(np.array(count_above_threshold), ax=axes[1, 1], cmap='viridis', square=True,
                        cbar_ax=fig.add_axes([0.903, 0.11, 0.015, 0.35]))
            axes[1, 1].set_title('Count Above 0.75 * Max Value')
            axes[1, 1].set_xticks(xticks)
            axes[1, 1].set_yticks(yticks)

            # Save the plots to file
            frame_dir = os.path.join(os.path.dirname(__file__), 'Base_Block_MPI_noGradAve_weights')
            os.makedirs(frame_dir, exist_ok=True)

            # Correct the frame filename
            frame_filename = os.path.join(frame_dir, f"epoch_{epoch}.png")
            plt.savefig(frame_filename)
            plt.close(fig)
        else:
            print('No weights found for output_projection')
    else:
        print('No output_projection key found in filtered weights')


def compare_datasets(saved, loaded, dataset_name):
    """
    Compares two datasets and logs mismatches.

    This function compares two datasets (`saved` and `loaded`) to check for mismatches. It handles
    both MLX-specific arrays (if `mx.array` is used) and generic arrays, using `np.array_equal` to
    verify if the datasets match. If a mismatch is found, it logs an error message and returns `False`.
    If the datasets match, it returns `True`.

    Parameters
    ----------
    saved : mx.array or numpy.ndarray
        The saved dataset that serves as the reference.
    loaded : mx.array or numpy.ndarray
        The loaded dataset to be compared against the saved dataset.
    dataset_name : str
        A string representing the name of the dataset being compared, used for logging purposes.

    Returns
    -------
    bool
        `True` if the datasets match, `False` otherwise.
    """
    # Compare MLX-specific arrays by converting them to numpy arrays
    if isinstance(saved, mx.array) and isinstance(loaded, mx.array):
        if not np.array_equal(np.array(saved), np.array(loaded)):
            print(f"Dataset {dataset_name} does not match.")
            return False
    else:
        # Compare generic numpy arrays or other data structures
        if not np.array_equal(saved, loaded):
            print(f"Dataset {dataset_name} does not match.")
            return False
    return True


def print_fresh_run_config(current_config):
    """
    Prints the current configuration for fresh runs to check the setup before starting.
    Each "Config Key" is printed on its own line, followed by the value for "Current Config".
    This is useful to inspect the configuration before a fresh run begins.

    Parameters
    ----------
    current_config : dict
        The configuration to print, including model parameters, training settings, etc.
    """
    print(" ===== Current Configuration for Fresh Run ====")
    print(" ----  Ensure everything is correct before starting ----")
    print(f"{'Config Key':<25}")
    print("=" * 50)

    # Print each key-value pair in the current configuration
    for key in sorted(current_config.keys()):
        current_value = current_config.get(key, "N/A")

        # Print the key and its value
        print(f"{key:<25}")
        print(f"  Current Config: {str(current_value)}")
        print("-" * 50)


def print_config_comparison(current_config, loaded_config):
    """
    Prints the current configuration alongside the reloaded configuration (from the checkpoint)
    for easy comparison. Each "Config Key" is printed on its own line, followed by the values for
    "Current Config" and "Loaded Config" on two separate, indented lines.
    """
    print(" ===== Comparison of Current Config and Loaded Config ====")
    print(" ----  Current config is NOT overwritten. Loaded Config is only used to check consistency ----")
    print(f"{'Config Key':<25}")
    print("=" * 50)

    # Get all unique keys from both configs
    all_keys = set(current_config.keys()).union(set(loaded_config.keys()))

    for key in sorted(all_keys):
        current_value = current_config.get(key, "N/A")  # Get value from current config or "N/A" if not present
        loaded_value = loaded_config.get(key, "N/A")  # Get value from loaded config or "N/A" if not present

        # Print the key and values on separate lines
        print(f"{key:<25}")
        print(f"  Current Config: {str(current_value)}")
        print(f"  Loaded Config:  {str(loaded_value)}")
        print("-" * 50)

# Convert lists to tuples for comparison purposes
def convert_lists_to_tuples(config):
    """
    Recursively converts lists to tuples in the configuration for consistency.
    This is particularly useful for configurations that expect tuples, and it ensures
    that comparison between current and loaded configurations is accurate.
    """
    if isinstance(config, dict):
        # Iterate over all keys in the dict
        for key, value in config.items():
            if isinstance(value, list):
                config[key] = tuple(value)
            elif isinstance(value, dict):
                convert_lists_to_tuples(value)  # Recurse for nested dicts
    return config


if __name__ == "__main__":
    """
    Main entry point for initializing geometry, boundary conditions, datasets, model, and optimizer, followed by
    training, evaluation, and saving the model.

    Steps:
    1. Initialize geometry and boundary conditions using the provided configuration.
    2. Generate training, validation, and test datasets, saving them to files if `start_from_scratch` is enabled.
    3. Initialize the model and optimizer from scratch or reload from a previous checkpoint.
    4. Define and compile the training and evaluation steps using MXNet.
    5. Train the model over multiple epochs, saving checkpoints at regular intervals.
    6. Plot predictions after training if enabled in the configuration.
    7. Optionally evaluate the model in a self-regressive manner and generate prediction plots.
    8. Evaluate the model on the test dataset.
    9. Optionally generate prediction plots and save the model and hyperparameters.

    Configurations:
    - Training parameters such as batch size, epochs, learning rate schedule, and geometry settings are configured in the `config` dictionary.
    - Dataset parameters like boundary conditions and thermal diffusivity values are initialized based on the configuration.
    - The training and validation loss are printed at the end of each epoch.
    - Model weights, optimizer state, random state, and hyperparameters are saved in the specified directory if saving is enabled.

    Parameters:
    - `config`: dict
        A dictionary containing the full configuration for the run, including model parameters, geometry settings,
        boundary conditions, batch size, learning rate, number of epochs, and whether to start from scratch or
        load from a checkpoint.
    - `hostname`: str
        The hostname of the machine, used for logging and debugging purposes.
    - `dataset_save_dir_path`: str
        The path to the directory where datasets are saved and loaded from.
    - `save_dir_path`: str
        The directory where checkpoints and model weights are saved.
    - `model_base_file_name`: str
        The base filename used for saving the model weights. The epoch number is appended to this base name.
    - `optimizer_base_file_name`: str
        The base filename used for saving the optimizer state. The epoch number is appended to this base name.
    - `hyper_base_file_name`: str
        The base filename used for saving the configuration (hyperparameters) at each checkpoint.

    Global Variables:
    - `training_data_mlx`, `validation_data_mlx`, `test_data_mlx`: Arrays containing the respective datasets.
    - `training_alphas_mlx`, `validation_alphas_mlx`, `test_alphas_mlx`: Thermal diffusivity values for each dataset.
    - `training_dts_mlx`, `validation_dts_mlx`, `test_dts_mlx`: Time steps for each dataset.
    - `learning_rate_schedule`: A dictionary defining learning rate adjustments at specific epochs.
    - `state`: A list containing the model state, optimizer state, and random state needed for model compilation.

    Returns:
        None.
    """
    print(f'Random state: {mx.random.state}')
    print(f'Configuration: {config["boundary_segment_strategy"]}:{config["model_params"]["mask_type"]}')

    # Print environment variables for debugging
    LIBUSED = os.environ.get('DYLD_LIBRARY_PATH')
    print(f"Hostname: {hostname}: DYLD_LIBRARY_PATH:{LIBUSED}")

    if config["start_from_scratch"]:  # This is a fresh run, create new datasets, save them to file and check
        nx, ny, training_bcs, validation_bcs, test_bcs, training_alphas, validation_alphas, test_alphas = initialize_geometry_and_bcs(
            config)
        (training_data_mlx, training_alphas_mlx, training_dts_mlx, validation_data_mlx, validation_alphas_mlx,
         validation_dts_mlx, test_data_mlx, test_alphas_mlx, test_dts_mlx) = generate_datasets(
            config, training_bcs, validation_bcs, test_bcs, training_alphas, validation_alphas, test_alphas)

        save_datasets(training_data_mlx, training_alphas_mlx, training_dts_mlx, validation_data_mlx,
                      validation_alphas_mlx, validation_dts_mlx, test_data_mlx, test_alphas_mlx, test_dts_mlx,
                      dataset_save_dir_path)

        (training_data_mlx_loaded, training_alphas_mlx_loaded, training_dts_mlx_loaded, validation_data_mlx_loaded,
         validation_alphas_mlx_loaded, validation_dts_mlx_loaded, test_data_mlx_loaded, test_alphas_mlx_loaded,
         test_dts_mlx_loaded) = load_datasets(
            dataset_save_dir_path)

        # Compare datasets to make sure that they were saved correctly and match when reloaded
        datasets_match = True
        datasets_match &= compare_datasets(training_data_mlx, training_data_mlx_loaded, "training_data_mlx")
        datasets_match &= compare_datasets(training_alphas_mlx, training_alphas_mlx_loaded, "training_alphas_mlx")
        datasets_match &= compare_datasets(training_dts_mlx, training_dts_mlx_loaded, "training_dts_mlx")
        datasets_match &= compare_datasets(validation_data_mlx, validation_data_mlx_loaded, "validation_data_mlx")
        datasets_match &= compare_datasets(validation_alphas_mlx, validation_alphas_mlx_loaded, "validation_alphas_mlx")
        datasets_match &= compare_datasets(validation_dts_mlx, validation_dts_mlx_loaded, "validation_dts_mlx")
        datasets_match &= compare_datasets(test_data_mlx, test_data_mlx_loaded, "test_data_mlx")
        datasets_match &= compare_datasets(test_alphas_mlx, test_alphas_mlx_loaded, "test_alphas_mlx")
        datasets_match &= compare_datasets(test_dts_mlx, test_dts_mlx_loaded, "test_dts_mlx")

        if datasets_match:
            print("All generated and saved datasets match.")
        else:
            print("Some datasets do not match.")
    else:
        # Calculate derived parameters
        (training_data_mlx, training_alphas_mlx, training_dts_mlx, validation_data_mlx, validation_alphas_mlx,
         validation_dts_mlx, test_data_mlx, test_alphas_mlx, test_dts_mlx) = load_datasets(
            dataset_load_dir_path)
        _, _, ny, nx = training_data_mlx.shape


    print(f"training data shape: {training_data_mlx.shape}")
    print(f"validation data shape: {validation_data_mlx.shape}")

    # Initialize model and optimizer
    model, optimizer = initialize_model_and_optimizer(config, nx, ny)
    model.eval()

    if not config["start_from_scratch"]:  # Reload from a previous checkpoint
        checkpoint_epoch = config["checkpoint_epoch"]
        start_epoch, loaded_optimizer_state, loaded_random_state, loaded_parameters, loaded_config = load_model_and_optimizer(model,
                                                                                                       optimizer,
                                                                                                       load_dir_path,
                                                                                                       model_base_file_name,
                                                                                                       optimizer_base_file_name,
                                                                                                       hyper_base_file_name,
                                                                                                       checkpoint_epoch)
        if loaded_config:
            # Convert lists to tuples in the loaded config for comparison purposes
            loaded_config = convert_lists_to_tuples(loaded_config)

            # Print side-by-side comparison of current and loaded config
            print_config_comparison(config, loaded_config)

        model.update(parameters=loaded_parameters)

        print(f'Random state before reloading: {mx.random.state}')
        mx.random.state = loaded_random_state
        print(f'Random state after reloading: {mx.random.state}')
        optimizer.init(model.trainable_parameters())
        optimizer.state = loaded_optimizer_state
        print("Current optimizer state after reloading:")
        print(f'   optimizer.betas: {optimizer.betas}')
        print(f'   optimizer.eps: {optimizer.eps}')
        print(f'   optimizer.step: {optimizer.step}')

        if compare_dict_states(optimizer.state, loaded_optimizer_state, 'optimizer state'):
            print('After reload: Optimizer state checks')
        else:
            print("After reload: Optimizer state mismatch detected.")

        if compare_list_states(mx.random.state, loaded_random_state, 'random state'):
            print("After reload: Random state checks.")
        else:
            print("After reload: Random state mismatch detected.")

        if compare_dict_states(model.parameters(), loaded_parameters, 'model state'):
            print("After reload: Model state checks.")
        else:
            print("After reload: Model state mismatch detected.")


    else:  # Start fresh run from scratch
        start_epoch = 0
        print_fresh_run_config(config)

    # Define state that will be needed by mx.compile
    state = [model.state, optimizer.state, mx.random.state]
    mx.eval(state)

    dx = config["geometry"]["dx"]
    dy = config["geometry"]["dy"]

    # Set the model to training mode
    model.train()


    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(src, target, src_alphas, src_dts, dx, dy):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn_2D)
        loss, grads = loss_and_grad_fn(model, src, target, src_alphas, src_dts, dx, dy)
        optimizer.update(model, grads)
        return loss, grads


    @partial(mx.compile, inputs=state, outputs=state)
    def evaluate_step(src, target, src_alphas, src_dts, dx, dy):
        loss = loss_fn_2D(model, src, target, src_alphas, src_dts, dx, dy)
        return loss


    print(f"************************ Hostname: {hostname} Starting Training *********************** ")

    # Start or continue training
    train_and_validate(training_data_mlx, training_alphas_mlx, training_dts_mlx, validation_data_mlx,
                       validation_alphas_mlx, validation_dts_mlx, config["model_params"]["batch_size"],
                       config["model_params"]["epochs"], start_epoch, config["save_interval"],
                       dx=config["geometry"]["dx"],
                       dy=config["geometry"]["dy"])

    model.eval()

    evaluate_model(model, data_loader_2D, test_data_mlx, test_alphas_mlx, test_dts_mlx, config["geometry"]["dx"],
                   config["geometry"]["dy"], config["model_params"]["batch_size"])

    if config["model_params"]["mask_type"] == "causal":
        # Call the autoregressive evaluation
        evaluate_self_regressive_model_BeyondL(model, data_loader_2D, test_data_mlx, test_alphas_mlx, test_dts_mlx,
                                               config["geometry"]["dx"],
                                               config["geometry"]["dy"], config["model_params"]["batch_size"],
                                               config["model_params"]["time_steps"],
                                               output_dir_regress=inference_mse_dir_path)


        if io_and_plots_config["plots"]["movie_frames"]:
            plot_regressive_predictions_2D(model, data_loader_2D, {'data': test_data_mlx, 'alphas': test_alphas_mlx,
                                                    'solution_dts': test_dts_mlx, 'batch_size': config["model_params"]["batch_size"],
                                                                   'shuffle': True},
                            num_examples=io_and_plots_config["plots"]["num_examples"],
                            output_dir=frameplots_save_dir_path,
                            t_trained=None)

    elif config["model_params"]["mask_type"] == "block":
        # Call the full sequence evaluation (non-autoregressive)
        evaluate_model_block_sequence(model, data_loader_2D, test_data_mlx, test_alphas_mlx, test_dts_mlx,
                                      config["geometry"]["dx"],
                                      config["geometry"]["dy"], config["model_params"]["batch_size"],
                                      config["model_params"]["time_steps"],
                                      output_dir_block=inference_mse_dir_path)

        if io_and_plots_config["plots"]["movie_frames"]:
            plot_predictions_2D(model, data_loader_2D, {'data': test_data_mlx, 'alphas': test_alphas_mlx,
                                                    'solution_dts': test_dts_mlx, 'batch_size': config["model_params"]["batch_size"],
                                                        'shuffle': True},
                            num_examples=io_and_plots_config["plots"]["num_examples"],
                            output_dir=frameplots_save_dir_path,
                            t_trained=None)

    if io_and_plots_config["model_saving"]:
        save_model_and_optimizer(config, config["model_params"]["epochs"], save_dir_path, model_base_file_name,
                                 optimizer_base_file_name, hyper_base_file_name, )
