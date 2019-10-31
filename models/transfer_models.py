from keras.applications import VGG16
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras import Model

from models import image_preprocessing
import numpy as np
import os


def vgg_model(attn):
    """
    Generate vgg model

    :param attn: bool flag for attention model encoding
    :return: return encoder Model instance
    """
    vgg_instance = VGG16(include_top=True, weights='imagenet')
    if attn:
        transfer_layer = vgg_instance.get_layer('block5_conv3')
    else:
        transfer_layer = vgg_instance.get_layer('fc2')
    vgg_transfer_model = Model(inputs=vgg_instance.input, outputs=transfer_layer.output)
    input_layer = vgg_instance.get_layer('input_1')
    image_size = input_layer.input_shape[1:3]
    return vgg_transfer_model, image_size


def use_pretrained_model_for_images(filenames_with_all_captions, attn, batch_size=64):
    """
    Uses the pretrained model without prediction layer to encode the images into the set of the features.

    :param filenames_with_all_captions: list of dictionaries containing images with the corresponding captions
    :param attn: bool flag for attention model encoding
    :param batch_size: size of the batch for CNN
    :return: np array with generated features
    """
    transfer_model, img_size = vgg_model(attn)
    # get the number of images in the dataset
    num_images = len(filenames_with_all_captions)
    # calculate the number of iterations
    iter_num = int(num_images / batch_size)
    # variable to print the progress each 5% of the dataset
    five_perc = int(iter_num * 0.05)
    iter_count = 0
    cur_progress = 0

    # get the paths to all images without captions
    image_paths = list(filenames_with_all_captions.keys())
    # list for the final result
    transfer_values = []

    # start and end index for each batch
    first_i = 0
    last_i = batch_size

    # loop through the images
    while first_i < num_images:
        iter_count += 1

        # progress print
        if iter_count == five_perc:
            iter_count = 0
            print(str(cur_progress) + "% of images processed")
            cur_progress += 5

        # to make sure that last batch is not beyond the number of the images
        if last_i > num_images:
            last_i = num_images

        # initialize the list for the batch
        image_batch = []

        # loop to form batches
        for image in image_paths[first_i:last_i]:
            # preprocess image
            image = image_preprocessing.image_preprocessing(image, img_size)
            # append image to batch list
            image_batch.append(image)

        # run the model to encode the features
        preds = transfer_model.predict(np.array(image_batch))

        # append predictions from the batch to the final list
        for pred in preds:
            transfer_values.append(pred)

        # update first and last indices in the batch
        first_i += batch_size
        last_i += batch_size

    reset_keras()
    del transfer_model
    return np.array(transfer_values)


def save_features(np_arr, folder, filename):
    """
    Saves encoded features into the .npy file.

    :param np_arr: the array with features which should be saved
    :param folder: path to the destination folder
    :param filename: filename of the features file
    """
    # form the full path for the file
    full_path = os.path.join(folder, filename)
    # create the folder if it does not exist
    if not os.path.exists(folder):
        os.mkdir(folder)
    # save file
    np.save(full_path, np_arr)
    print("Array was saved to {}".format(full_path))


def reset_keras():
    """
    Releases keras session
    """
    sess = get_session()
    clear_session()
    sess.close()
