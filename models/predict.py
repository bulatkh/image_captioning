from keras import Model
from tqdm import tqdm_notebook as tqdm

from models import image_preprocessing
import math
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np


def generate_caption(image, model_image_size, decoder_model, transfer_model, vocabulary, transfer_values=False,
                     beam_size=3, max_caption_len=30, attn=False, get_weights=False):
    """
    Generates captions for a given image

    :param image: image path/array with transfer values
    :param model_image_size: size for the encoder model
    :param decoder_model: instance of the decoder model for prediction
    :param transfer_model: instance of the encoder model for prediction
    :param vocabulary: vocabulary used for training the decoder model
    :param transfer_values: bool flag for transfer values
    :param beam_size: size of a beam for predictions
    :param max_caption_len: maximum length of captions
    :param attn: bool flag for using attention mechanism
    :param get_weights: bool flag for getting attention weights
    :return: predicted captions + attention weights if get_weights param is true
    """
    if transfer_values:
        if attn:
            input_transfer_values = image.reshape((1, 196, 512))
        else:
            input_transfer_values = image.reshape((1, 4096))
    else:
        img = image_preprocessing.image_preprocessing(image, model_image_size)
        image_batch = np.expand_dims(img, axis=0)
        input_transfer_values = transfer_model.predict(image_batch)
        if attn:
            input_transfer_values = np.reshape(input_transfer_values, [-1, 196, 512])

    decoder_inputs = np.zeros(shape=(1, max_caption_len), dtype=np.int)
    captions = [[[vocabulary.get_id_by_word('<sos>')], 0.0]]

    weights = []
    for i in range(max_caption_len - 1):
        tmp_caps = []
        for caption in captions:
            sentence, score = caption
            if sentence[-1] == vocabulary.get_id_by_word('<eos>'):
                tmp_caps.append(caption)
                continue

            decoder_inputs[0, :len(sentence)] = sentence

            input_data = {
                'encoder_input': input_transfer_values,
                'decoder_input': decoder_inputs
            }

            decoder_output = decoder_model.predict(input_data)

            candidates = decoder_output[0, i, :].argsort()[-beam_size:]

            for candidate in candidates:
                sentence.append(candidate)
                caption = [sentence, score + np.log(decoder_output[0, i, candidate])]
                sentence = sentence[:-1]
                tmp_caps.append(caption)

        captions = sorted(tmp_caps, key=lambda x: x[1], reverse=True)[:beam_size]

    for i in range(beam_size):
        captions[i][1] /= len(captions[0][0])

    captions = sorted(tmp_caps, key=lambda x: x[1], reverse=True)

    if attn and get_weights:
        outputs = []
        for i in range(len(captions[0][0])):
            attn_weight_layer = decoder_model.get_layer('weights_{}'.format(i)).output
            outputs.append(attn_weight_layer)
        decoder_inputs[0, :len(captions[0][0])] = captions[0][0]
        input_data = {
            'encoder_input': input_transfer_values,
            'decoder_input': decoder_inputs
        }
        attn_weight_model = Model(inputs=decoder_model.inputs, outputs=outputs)
        weights = attn_weight_model.predict(input_data)
    res_captions = []
    probs = []
    for caption in captions[:beam_size]:
        res_captions.append([vocabulary.get_word_by_id(x) for x in caption[0][1:-1]])
        probs.append(caption[1])
    if attn and get_weights:
        return res_captions, probs, weights
    else:
        return res_captions, probs


def generate_test_captions(test_images, *args):
    """
    generates captions for the given set of images

    :param test_images: array with paths to test images
    :param args: arguments for nested function
    :return: captions list
    """
    captions = []
    for i, image in tqdm(enumerate(test_images)):
        captions.append(generate_caption(image, *args))

    return captions


def transform_captions(captions):
    """
    Transforms input captions where each caption is an array of words to string representation
    :param captions: list of array-like captions
    :return: list of string captions
    """
    res_captions = []
    for caption in captions:
        caption[0] = caption[0].capitalize()
        res_captions.append(' '.join(caption) + '.')
    return res_captions


def get_weights_plot(best_caption, weights, path, mode, save_path=''):
    """
    Generates and saves or shows the attention weights visualisation

    :param best_caption: caption for which the visualisation is going to be generated
    :param weights: attention weights array
    :param path: path to the original image
    :param mode: save or show mode
    :param save_path: destination path of the visaulisation
    """
    cols = 4
    rows = math.ceil(len(best_caption) / cols)
    plt.figure(1, figsize=(12, 12))
    for word_num in range(len(best_caption)):
        print(best_caption[word_num])
        weights_img = np.reshape(weights[word_num], [14, 14])
        weights_img = misc.imresize(weights_img, (224, 224))
        img = image_preprocessing.image_preprocessing(path, (224, 224))
        plt.subplot(rows, cols, word_num + 1)
        plt.title(best_caption[word_num], fontsize=20)
        plt.imshow(img)
        plt.imshow(weights_img, cmap='bone', alpha=0.8)
        plt.axis('off')
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    if mode == 'save':
        plt.savefig(save_path)
    elif mode == 'show':
        plt.show()
    else:
        raise ValueError('Please select the valid mode')
    plt.clf()
