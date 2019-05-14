import json
import numpy as np
import os
from matplotlib.pyplot import imshow
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


def get_image_filename_with_caption(captions_path, images_path, train=True):
    """
    Parses JSON file and returns list with dictionaries of the following format
    {id, path to the image, a caption}
    
    Parameters:
    -----------
    captions_path: str
        Path to the folder with caption file
        
    images_path: str
        Path to the folder with images
        
    train: boolean
        Flag for the training dataset
    -----------
    """
    if train:
        full_path = os.path.join(captions_path, 'captions_train2017.json')
    else:
        full_path = os.path.join(captions_path, 'captions_val2017.json')

    with open(full_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    images = data['images']
    annotations = data['annotations']

    img_dict = {}

    for img in images:
        img_dict.update({img['id']: img['file_name']})

    ann_dict = {}
    for ann in annotations:
        ann_dict.update({ann['caption']:ann['image_id']})

    
    res = []

    for key, val in ann_dict.items():
        if train:
            img_path = os.path.join(images_path, 'train2017', 'train2017', img_dict[val])
        else:
            img_path = os.path.join(images_path, 'val2017', 'val2017', img_dict[val])
            
        res.append({'id':val, 'path': img_path, 'caption':key})
        
    return res


def show_image_with_captions_by_id(idx, filenames_with_captions):
    """
    Function returns an image from the train/val dataset with all the captions by id of the image
    
    Parameters
    -----------
    idx : int
        ID of the requested image
        
    filenames_with_captions
        List of dictionaries containing images with the corresponding captions
    ----------
    """
    captions = []
    path = ''
    for file in filenames_with_captions:
        if file['id'] == idx:
            captions.append(file['caption'])
            path = file['path']
    assert path != ''
    
    for cap in captions:
        print(cap)
        
    image = Image.open(path)
    imshow(np.asarray(image))


def show_image_with_caption(num, filenames_with_captions):
    """
    Returns image of the given number in the list of dictionaries
    
    Parameters
    -----------
    num: int
        number of returned image in the list
    
    filenames_with_captions
        List of dictionaries containing images with the corresponding captions
    -----------
    """
    path = filenames_with_captions[num]['path']
    idx = filenames_with_captions[num]['id']
    caption = filenames_with_captions[num]['caption']
    print(idx)
    print(caption)
        
    image = Image.open(path)
    imshow(np.asarray(image))

    
def get_image_with_all_captions(filenames_with_captions):
    """
    Function returns a dictionary of the following format
    {image path, all captions related to the image}
    
    Parameters
    -----------
    filenames_with_captions
        List of dictionaries containing images with the corresponding captions
    -----------
    """
    image_path_with_captions = dict()
    for image in filenames_with_captions:
        image_path_with_captions.update({image['path']:[]})
    for image in filenames_with_captions:
        tmp_captions = image_path_with_captions[image['path']]
        tmp_captions.append(image['caption'])
        image_path_with_captions.update({image['path']:tmp_captions})  
    return image_path_with_captions

def make_list_of_captions(filenames_with_all_captions):
    """
    Extracts captions from the list of dictionaries with filenames and captions
    
    Parameters:
    -----------
    filenames_with_all_captions: list
        List of dictionaries containing images with the corresponding captions
    -----------
    """
    captions = []
    for _, val in filenames_with_all_captions.items():
        captions.append(val)
    return captions