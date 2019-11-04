# Image Captioning using Encoder-Decoder Framework and Attention Mechanism
## Project Summary
This repository contains code for the Image Captioning Master's Thesis Project for the MSc Advanced Computer Science at The University of Manchester. All the implemented Image Captioning models were built using TensorFlow and keras on Flickr8k dataset (the smallest labelled dataset for Image Captioning) due to the available environment. Nevertheless, the existing architecture might be used for the bigger datasets (e.g. the module for parsing MS COCO dataset was also developed). 

All the implemented models follow the Encoder-Decoder framework and some of them also use Attention mechanisms. The designed experiment consists of two main stages: hyperparameter tuning based for the baseline encoder-decoder (VGG16-GRU) model and comparison of different decoders. In the first stage of the experiment, nine models with different hyperparameters were trained and tested for captions precision, time and memory efficiency. The models from the second stage experiments include decoders based on two different Recurrent Neural Networks architectures (GRU and LSTM) with regularisation and two attention mechanisms. These models were assessed and analysed according to a specific evaluation framework which takes into account various aspects of model performance including captions precision (BLEU scores), lexical richness, quantitative and qualitative analysis of the beam search predicting algorithm and comparison of the attention mechanisms.

## Data Preprocessing
Data Preprocessing steps include the following processes:
1. Parse and sample a dataset. Modules ```datasets/flickr8k_parse.py``` and ```datasets/coco_parse.py``` and were developed to parse Flickr8k and MS COCO datasets (including images and annotation files), respectively. Using the modules, ```<image: list_of_captions>``` pairs might be obtained. The modules should be used to obtain training, validation and test data independently 
2. The vocabulary has to be built based on the training set. The vocabulary class is available in ```text_preprocessing.py```.
3. Finally, the batch generation algorithm should be defined. In this project, a single batch contains randomly selected ```<image-caption>``` pairs. The functions for batch generation might be obtained from ```models/batch_generator.py```.

## Implemented Models
Pre-trained VGG16 (```models/encoder.py```) was used as the encoder for all the architectures with some minor changes, namely features were extracted from a fully-connected layer in case of models without attention and from the last convolution block for the models with attention. The encoder saves image features in .npy files which later might be used in a decoder.

On the contrary, various types of decoders (```models/decoder.py```) were implemented and compared in the projects, such as:
* GRU decoder without regularisation (baseline). The baseline model was tested with different hyperparameters including batch size and number of recurrent layers to find the most suitable values of them for Flickr8k data.
* GRU decoder with Batch Normalisation and Dropout regularisation,
* LSTM decoder with regularisation,
* LSTM decoder with regularisation and scaled dot-product attention.
* LSTM decoder with regularisation and soft attention.

In addition, the models generated predictions using beam search (with beam sizes 1, 3 and 5). The predicting algorithm is implemented in ```models/predict.py```.

## Results and Evaluation
This section presents the main findings and results while the full description of the developed evaluation framework and the complete analysis of the models is available in the body of the Dissertation.

### Quantitative Results
The BLEU-scores obtained in the project for Flickr8k dataset are lower than the ones in the related papers (e.g. [Xu et al., 2015](https://arxiv.org/pdf/1502.03044.pdf)). That might be mostly related to some differences in hyperparameters and specific regularisation terms in the loss function. The results on the test sample for all the models with beam size 3 are presented in the table below. All the models had 2-layer decoders and used batch size 32.

| Decoder | Regularisation | Attention | Attention type     | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---------|----------------|-----------|--------------------|--------|--------|--------|--------|
| GRU     | -              | -         | -                  | 50.928 | 31.141 | 19.375 | 12.423 |
| GRU     | +              | -         | -                  | 52.191 | 32.815 | 20.66  | 12.618 |
| LSTM    | +              | -         | -                  | 53.881 | 34.339 | 21.973 | 14.05  |
| LSTM    | +              | +         | soft (Bahdanau)    | 55.174 | 35.941 | 23.557 | 15.013 |
| LSTM    | +              | +         | scaled dot-product | 55.322 | 35.757 | 22.946 | 14.32  |

With the beam size 5, the BLEU-scores for the best model were improved as follows:
* BLEU-1: 56.877
* BLEU-2: 36.407
* BLEU-3: 23.893
* BLEU-4: 15.182

In addition, the lexical richness of the generated captions for test images was assessed. It turned out that regularisation techniques significantly enriches the active vocabulary of the models.

### Qualitative Results
The qualitative analysis includes a comparison of various attention mechanisms in terms of how they attend an image. The following figures show the difference between visualisations for scaled dot-product attention and soft attention.

![football_att](/figures/football_att.png)
Figure 1: Visualisation of the attention mechanisms for the image with football players: (left) - soft attention; (right) - scaled dot-product attention.

![dog_att](/figures/brown_dog_att.png)
Figure 2: Visualisation of the attention mechanisms for the image with a running dog: (left) - soft attention; (right) - scaled dot-product attention.

![bicycle_att](/figures/bicycle_att.png)
Figure 3: Visualisation of the attention mechanisms for the image with a cyclist: (left) - soft attention; (right) - scaled dot-product attention.

## Image Captioning System
A simple prototype of a web application for image captioning was implemented using Flask framework. The system allows a user to attach an image or use one of several default images, generate captions for the image with specified beam size and check the visualisation of the attention mechanism. The model used in the app is the one with LSTM decoder, regularisation and soft attention (the model with the best performance on test data).

A typical use case of the implemented application is shown in Figure 4.

![bicycle_att](/figures/use_case1.png)
Figure 4: A use case of the system.

## How To Use
The project uses [YACS library](https://github.com/rbgirshick/yacs) for managing configurations. The architectures and hyperparameters are managed using ```configs``` directory: ```configs/default.py``` contains a default set of hyperparameters. Each notebook includes the following code to conduct experiments for the baseline model with another set of hyperparameters defined in ```configs/baseline.yaml```:
```
config_file = "./configs/baseline.yaml"
update_config(config, config_file)
```
So, another experiment might be launched similarly by creating a yaml file and passing its path to ```config_file``` variable.

The following steps are the recommendations on how to use the code:
1. Launch ```encoder.ipynb``` and run all cells to encode image features using VGG16 encoder network. Please, note that you should specify the path to the dataset in the configurations file. Also, features for models with attention and without it are taken from the different layers of the network.
2. Run all cells in ```train_model.ipynb``` to train the model from your configurations.
3. The model might be tested in ```evaluation.ipynb```.

To run the app you should:
1. Perform steps 1 and 2 from the above list using ```configs/attn.yaml``` configurations file. TThis file contains the configuration for the model with soft attention showed the best performance among all the implemented models. You should alter cells with configuration updates to the following:
```
config_file = "./configs/attn.yaml"
update_config(config, config_file)
```
2. Activate virtual environment. For Windows:
```
.\venv\Scripts\activate
```
3. Run flask application:
```
flask run
```
4. After several initialisations, you will see the local address. Copy and paste it to your browser line to use the app.

**Note:** The code in the repository is not well tested for COCO dataset although there is a module to parse it which works generally fine. Thus, errors might be raised while using COCO dataset for model training, testing or evaluation.
