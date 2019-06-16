from PIL import Image
import numpy as np

def generate_model_path(gru, layers, batch_norm, drop):
    if gru:
        model = 'GRU'
    else:
        model = 'LSTM'
    if batch_norm:
        bn = '_bn'
    else:
        bn = ''
    if drop:
        dr = '_dr'
    else:
        dr = ''
    path = './models/VGG16_{}_{}l{}{}.json'.format(model, layers, bn, dr)
    return path

def generate_weights_path(gru, dataset, layers, batch_size, batch_norm, drop):
    if gru:
        model = 'GRU'
    else:
        model = 'LSTM'
    if batch_norm:
        bn = '_bn'
    else:
        bn = ''
    if drop:
        dr = '_dr'
    else:
        dr = ''
    path = './weights/VGG_{}_{}_{}l_{}b{}{}.hdf5'.format(gru, dataset, layers, batch_size, bn, dr)
    return path

def image_preprocessing(image_path, new_size):
    """
    Reads the image and applies preprocessing including:
    - resizing to the new_size
    - rescaling pixel values at [0, 1]
    - transforming grayscale images to RGB format
    
    Parameters:
    -----------
    image_path : str
        full path to the image
    new_size: tuple
        size of the output image
    -----------
    """
    image = Image.open(image_path)
    image = np.array(image.resize(new_size, Image.LANCZOS))
    image = np.divide(image, 255)
    if len(image.shape) != 3:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image

