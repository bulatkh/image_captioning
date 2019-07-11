from PIL import Image
import numpy as np


def image_preprocessing(image_path, new_size):
    """
    Reads the image and applies preprocessing including:
    - resizing to the new_size
    - rescaling pixel values at [0, 1]
    - transforming grayscale images to RGB format

    :param image_path: full path to the image
    :param new_size: tuple with size of the output image
    :return: preprocessed image
    """
    image = Image.open(image_path)
    image = np.array(image.resize(new_size, Image.LANCZOS))
    image = np.divide(image, 255)
    if len(image.shape) != 3:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image
