"""Helper methods for using Keras, often using keract

Yields:
    [type]: [description]
"""
from typing import Tuple

import keract
import numpy as np
from tqdm.auto import tqdm

from . import dbhandler
from . import neural_networks as nn
from .frame_extractor import FrameExtractor
from .image_preprocessing import process_image


def add_layer_activations_to_database(
    network: nn.Network, database_file: str, frames: FrameExtractor
) -> Tuple[dict, dict]:
    """Populates the database with with he layer from all images described by the
    FrameExtractor.

    The specific neural network and layer to use is described by the Network class.

    Args:
        network (nn.Network): The Network class describing which network and layer to use
        database_file (str): Where to find the database file
        frames (FrameExtractor): The thumnbnail generator, which includes a list of all frames

    Returns:
        Tuple[dict, dict]: [description]
    """
    used_model_layer = network.default_layer

    with dbhandler.VRDDatabase(database_file) as dbc:
        for img in tqdm(frames.all_images):
            db_activation = dbc.get_layer_data(img)
            if db_activation is None:
                activations = keract.get_activations(
                    network.used_model,
                    process_image(img, network.target_size, trim=True),
                    auto_compile=True,
                )
                activations_keys = list(activations.keys())

                selected_activation = activations_keys[used_model_layer]
                layer_data = activations[selected_activation]
                value = np.reshape(layer_data, (1, np.product(layer_data.shape)))
                dbc.add_layer_data(img, value)

    # return results_dict, results_dict_full


# as per https://stackoverflow.com/a/8290508
def batch(iterable, batch_size=1):
    """Split a long list into separate batches according to batch_size"""
    iter_length = len(iterable)
    for ndx in range(0, iter_length, batch_size):
        yield iterable[ndx : min(ndx + batch_size, iter_length)]


def display_activation_for_image(filename: str, network: nn.Network):
    """Uses keract to display activation for a given image!

    Args:
        filename (str): The image
    """
    activations = keract.get_activations(
        network.used_model,
        process_image(filename, network.target_size, trim=True),
        auto_compile=True,
    )
    keract.display_activations(
        activations,
        cmap=None,
        save=False,
        directory=".",
        data_format="channels_last",
        fig_size=(24, 24),
        reshape_1d_layers=False,
    )
