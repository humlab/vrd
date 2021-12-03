"""Contains specification for the available neural networks for use in the VRD
"""
from enum import Enum
from dataclasses import dataclass
from typing import Any


from tensorflow.keras.applications import vgg16, resnet50, mobilenet, inception_v3


class NeuralNetworks(Enum):
    """Enum for the different Neural Networks available"""

    vgg16 = 1
    resnet50 = 2
    mobilenet = 3
    inception_v3 = 4


@dataclass
class Network:
    """Specifics on the netowork used.
    This includes:
        * Used network and model (e.g. VGG16 with imagenet weights)
        * Target size of images before prediction (Can be changed to some degree)
        * Which layer should be extracted  (Can be changed)
    """

    network_enum: Any
    used_network: Any
    used_model: Any
    target_size: tuple
    default_layer: int

    def __str__(self):
        return f"Neural network used: {self.used_network}\nModel used:{self.used_model}\nInput size: {self.target_size}\nLayer used: {self.default_layer}"


def get_network(network: NeuralNetworks) -> Network:
    """Gets a neural network including default settings

    Args:
        network (NeuralNetworks): The network to get, available in the NeuralNetworks enum

    Raises:
        Exception: If the enum is invalid

    Returns:
        Network: A Network class describing the desired network
    """

    if network is NeuralNetworks.inception_v3:
        return Network(
            network_enum=network,
            used_network=inception_v3,
            used_model=inception_v3.InceptionV3(weights="imagenet"),
            target_size=(299, 299),
            default_layer=100,
        )  # FIXME: Set tested default value

    if network is NeuralNetworks.resnet50:
        return Network(
            network_enum=network,
            used_network=resnet50,
            used_model=resnet50.ResNet50(weights="imagenet"),
            target_size=(224, 224),
            default_layer=165,
        )
    if network is NeuralNetworks.vgg16:
        return Network(
            network_enum=network,
            used_network=vgg16,
            used_model=vgg16.VGG16(weights="imagenet"),
            target_size=(224, 224),
            default_layer=100,
        )  # FIXME: Set tested default value
    if network is NeuralNetworks.mobilenet:
        return Network(
            network_enum=network,
            used_network=mobilenet,
            used_model=mobilenet.MobileNet(weights="imagenet"),
            target_size=(224, 224),
            default_layer=86,
        )
    raise Exception("No such network type!")  # Add valid exception
