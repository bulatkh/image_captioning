
# coding: utf-8

# In[1]:


import json
import numpy as np
import os
from matplotlib.pyplot import imshow
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def get_image_filename_with_caption(captions_path, images_path, train=True):
    """
    Function parses JSON file and returns list with dictionaries of the following format
    {id, path to the image, a caption}
    
    inputs:
    - captions_path - path to the folder with caption file
    - images_path - path to the folder with images
    - train - flag of the train dataset
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


# In[4]:


def show_image_with_captions_by_id(idx, filenames_with_captions):
    """
    Function returns an image from the train/val dataset with all the captions by id of the image
    
    idx - image id
    filenames_with_captions - list with dictionaries returned by the function above
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


# In[5]:


def show_image_with_caption(num, filenames_with_captions):
    """
    Function returns image of the number from the list of dictionaries
    
    num - number of returned image in the list
    filenames_with_captions - list with dictionaries returned by the function above
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
    {image path, all caption related to the image}
    
    filenames_with_captions - list with dictionaries returned by the function above
    """
    image_path_with_captions = dict()
    for image in filenames_with_captions:
        image_path_with_captions.update({image['path']:[]})
    for image in filenames_with_captions:
        tmp_captions = image_path_with_captions[image['path']]
        tmp_captions.append(image['caption'])
        image_path_with_captions.update({image['path']:tmp_captions})  
    return image_path_with_captions