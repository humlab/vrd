"""A file containting some helper files for the faiss library

"""
from typing import TYPE_CHECKING

import faiss
from tqdm.auto import tqdm
import numpy as np

from . import dbhandler, frame_extractor, neighbours, neural_networks

if TYPE_CHECKING:
    from .neighbours import Neighbours


def get_faiss_index(
    database_file: str,
    network: neural_networks.Network,
    frames: frame_extractor.FrameExtractor,
):
    """Creates a faiss index from all the frames in the specified FrameExtractor.
    This requires the database to be already populated with valid layer data

    Args:
        database_file (str): The database containing the layer data
        network (neural_networks.Network): The network used to create the data
        frames (frame_extractor.FrameExtractor): The frame extractor describing the included files

    Returns:
        A faiss index, filled with indexes
    """

    def get_layer_value_count(x):
        return np.asscalar(
            np.prod([i for i in x.output_shape if i])
        )  # TODO: Depricated asscalar

    layer_size = get_layer_value_count(network.used_model.layers[network.default_layer])

    faiss_index = faiss.IndexFlatL2(layer_size)

    with dbhandler.VRDDatabase(database_file) as dbc:
        for img in tqdm(frames.all_images):
            # Maybe more stable if we do batches instead?
            db_activation = dbc.get_layer_data(img)
            # Here we would add any (optional) preprocessing steps, e.g. centering or scaling of some sort
            faiss_index.add(db_activation)
    return faiss_index


# as per https://stackoverflow.com/a/8290508
def batch(iterable, batch_size=1):
    """Helper function to run a large number of iteratebles in batches"""
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx : min(ndx + batch_size, length)]


def _correct_indexes(distance_list):
    """Ensures that indexes are correctly sorted.

    This implies that the "source" frame is always at index 0.

    This can be incorrect in case there are duplicates in the faiss database.

    Args:
        distance_list ([type]): The distance list

    Returns:
        A distance list with corrected indexes
    """
    new_distance_list = []
    for expected_idx, d_i in enumerate(distance_list):
        d, i = d_i
        if i[0] != expected_idx:  # Couldn't find correct index in list
            correct_loc = list(np.nonzero(i == expected_idx)[0])
            if len(correct_loc) == 0:  # If not found
                d = np.insert(d, 0, 0)
                i = np.insert(i, 0, expected_idx)
            else:  # Correct index was found.
                # Swap locations. We assume distance is the same.
                i[correct_loc] = i[0]
                i[0] = expected_idx
        new_distance_list.append((d, i))
    return new_distance_list


def calculate_distance_list(
    frames: frame_extractor.FrameExtractor,
    database_file: str,
    faiss_index,
    neighbour_num=100,
    batch_size=1000,
):
    """Find the neighbour_num closest matches to each"""
    current_batch_start = 0
    all_images = frames.all_images
    # print('Opening DB....')
    with dbhandler.VRDDatabase(database_file) as dbc:
        distance_list = []
        for curr_batch in tqdm(
            batch(range(len(all_images)), batch_size),
            total=np.ceil(len(all_images) / batch_size),
        ):
            profiles = np.array(
                [dbc.get_layer_data(all_images[x]).flatten() for x in curr_batch]
            )
            d, i = faiss_index.search(profiles, neighbour_num)

            for combined in zip(d, i):
                distance_list.append(combined)

            current_batch_start += len(i)

    d_list = _correct_indexes(distance_list)

    # Sanity check for uniqueness
    actual_unique = np.unique([x[1][0] for x in d_list])
    if len(actual_unique) != len(distance_list):
        print(
            f"Warning: Incorrect number of unique indexes in distance list.\nExpected: {len(distance_list)}, got {len(actual_unique)}"
        )
    return neighbours.Neighbours(frames, distance_list)
