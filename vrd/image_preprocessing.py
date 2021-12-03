"""A module containing methods relating to preprocessing and modification of images
"""

import numpy as np
from PIL import Image as PILImage
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from wand.image import Image

from .neural_networks import Network


def trim_image(imgpath: str, fuzz=5000):
    """Trims the given image as per the trim command in ImageMagick
    ( https://imagemagick.org/index.php ).

    This is called using the wrapper wand ( https://docs.wand-py.org/ )

    Args:
        imgpath (str): A string path to the image file to trim
        fuzz (int, optional): The "fuzz" allowed, generally due to noise. . Defaults to 5000.

    Returns:
        An numpy array containing the image
    """
    img = Image(filename=imgpath)
    img.trim(fuzz=fuzz)
    return img


def process_image(filename, target_size, trim=False, trim_fuzz=5000):
    """Preproccess the image.

    This currently includes removing any single-color border (e.g. black border) around
    an image using the trim command, as well as rescale to correct target size.

    Code modified from code found online, the original source is possibly:
    https://medium.datadriveninvestor.com/product-recommendation-based-on-visual-similarity-on-the-web-machine-learning-project-end-to-end-6d38d68d414f

    Args:
        filename (string path): The path to the image to be processed
        target_size (tuple): The requested size (x,y) of the final image in pixels
        trim (bool, optional): Whether or not to perform the trim command
        trim_fuzz (int, optional): The "fuzz" to allow when trimming. Defaults to 5000.

    Returns:
        np.array: A numpy array containing the image
    """
    original = None
    if trim:
        original = trim_image(filename, fuzz=trim_fuzz)
        # If trimmed too hard, simply do not trim. This can cause issues later otherwise.
        # print(f'Size after trim: {original.size}')
        if np.max(original.size) < 30:
            # TODO: Log this? Image is too small, simply use the original... or skip?
            original = None
        else:
            original.resize(*target_size)

    if original is None:
        original = load_img(filename, target_size=target_size)

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)

    return image_batch


def predict_image(image_batch, network: Network):
    """Predict a batch of images

    Mostly depricated, as we do not currently utilize this method.

    Code modified from code found at:
    https://learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/

    Args:
        image_batch ([type]): [description]
        network (Network): [description]
    """
    processed_image = network.used_network.preprocess_input(image_batch.copy())
    predictions = network.used_model.predict(processed_image)
    labels = network.used_model.decode_predictions(predictions)

    for prediction_id in range(len(labels[0])):
        print(labels[0][prediction_id])


def is_monochrome(file, allowed_difference=40):
    """Determine if an image file is largely the same color.

    This is done by checking the range of colors, i.e. minimum and maximum,
    of all color channels. If there is never a difference larger than the
    allowed_difference argument, the image is assumed to be largely the same color
    and True is returned.


    Args:
        file (str): The path to the image file
        allowed_difference (int, optional): How much difference is allowed in the colors.
            Defaults to 40 (out of 255).

    Returns:
        bool: True if the image is largely the same color, otherwise False
    """
    img = PILImage.open(file)
    extrema = PILImage.Image.getextrema(img)
    for ext in extrema:
        if np.diff(ext) > allowed_difference:
            return False
    return True
