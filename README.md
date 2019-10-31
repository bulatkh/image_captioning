# Image Captioning using Encoder-Decoder Framework and Attention Mechanism
## Project Summary
This repository contains code for the Image Captioning Master's Thesis Project for the MSc Advanced Computer Science at The University of Manchester. All the implemented Image Captioning models were built on Flickr8k dataset (the smallest labelled dataset for Image Captioning) due to the available environment. Nevertheless, the existing architecture might be used for the bigger datasets (e.g. the module for parsing MS COCO dataset was also developed). 

All the implemented models follow the Encoder-Decoder framework and some of them also use Attention mechanisms. The designed experiment consists of two main stages: hyperparameter tuning based for the baseline encoder-decoder (VGG16-GRU) model and comparison of different decoders. In the first stage of the experiment, nine models with different hyperparameters were trained and tested for captions precision, time and memory efficiency. The models from the second stage experiments include decoders based on two different Recurrent Neural Networks architectures (GRU and LSTM) with regularisation and two attention mechanisms. These models were assessed and analysed according to a specific evaluation framework which takes into account various aspects of model performance including captions precision (BLEU scores), lexical richness, quantitative and qualitative analysis of the beam search predicting algorithm and comparison of the attention mechanisms.

## Data Preprocessing
Data Preprocessing steps include the following processes:
1. Parse and sample a dataset. Modules ```datasets/flickr8k_parse.py``` and ```datasets/coco_parse.py``` and were developed to parse Flickr8k and MS COCO datasets (including images and annotation files), respectively. Using the modules, ```<image: list_of_captions>``` pairs might be obtained. The modules should be used to obtain training, validation and test data independently 
2. The vocabulary has to be built based on the training set. The vocabulary class is available in ```text_preprocessing.py```.
3. Finally, the batch generation algorithm should be defined. In this project, a single batch contains randomly selected ```<image-caption>``` pairs. The functions for batch generation might be obtained from ```models/batch_generator.py```.

## Implemented Models
Pre-trained VGG16 (```models/encoder.py```) was used as the encoder for all the architectures with some minor changes, namely features were extracted from a fully-connected layer in case of models without attention and from the last convolution block for the models with attention. The encoder saves image features in .npy files which later might be used in a decoder.

On the contrary, various types of decoders (```models/decoder.py```) were implemented and compared in the projects, such as:
* GRU decoder without regularization (baseline). The baseline model was tested with different hyperparameters including batch size and number of recurrent layers to find the most suitable values of them for Flickr8k data.
* GRU decoder with Batch Normalization and Dropout regularisation,
* LSTM decoder with regularisation,
* LSTM decoder with regularisation and scaled dot-product attention.
* LSTM decoder with regularisation and soft attention.

In addition, the models generated predictions using beam search (with beam sizes 1, 3 and 5). 

## Results and Evaluation
### Quantitative Results
The BLEU-scores obtained in the project are lower than the ones in the related papers (e.g. [Xu et al., 2015](https://arxiv.org/pdf/1502.03044.pdf)). That might be mostly related to some differences in hyperparameters and specific regularization terms in loss function. The results for the best model 

### Qualitative Results

## Image Captioning System

## How To Use
