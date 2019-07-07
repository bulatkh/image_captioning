from PIL import Image
import numpy as np


def generate_model_path(gru, layers, batch_norm, drop, attention, attn_type):
    attn = ''
    dr = ''
    bn = ''
    if gru:
        model = 'GRU'
    else:
        model = 'LSTM'
        if attention:
            attn = '_attn_' + attn_type
    if batch_norm:
        bn = '_bn'
    if drop:
        dr = '_dr'
    
    path = './models/VGG16_{}_{}l{}{}{}.json'.format(model, layers, bn, dr, attn)
    return path


def generate_weights_path(gru, dataset, layers, batch_size, batch_norm, drop, attention, attn_type):
    attn = ''
    dr = ''
    bn = ''
    if gru:
        model = 'GRU'
    else:
        model = 'LSTM'
        if attention:
            attn = '_attn_' + attn_type
    if batch_norm:
        bn = '_bn'
    if drop:
        dr = '_dr'
    path = s.format(model, dataset, layers, batch_size, bn, dr, attn)
    return path


def generate_callback_path(gru, dataset, layers, batch_size, batch_norm, drop, attention, attn_type):
    attn = ''
    dr = ''
    bn = ''
    if gru:
        model = 'GRU'
    else:
        model = 'LSTM'
        if attention:
            attn = '_attn_' + attn_type
    if batch_norm:
        bn = '_bn'
    if drop:
        dr = '_dr'
    path = './callbacks/VGG16_{}_{}_{}l_{}b{}{}{}.csv'.format(model, dataset, layers, batch_size, bn, dr, attn)
    return path


def generate_captions_path(gru, dataset, layers, batch_size, batch_norm, drop, attention, attn_type):
    attn = ''
    dr = ''
    bn = ''
    if gru:
        model = 'GRU'
    else:
        model = 'LSTM'
        if attention:
            attn = '_attn_' + attn_type
    if batch_norm:
        bn = '_bn'
    if drop:
        dr = '_dr'
    path = './captions/VGG16_{}_{}_{}l_{}b{}{}{}.txt'.format(model, dataset, layers, batch_size, bn, dr, attn)
    return path


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
