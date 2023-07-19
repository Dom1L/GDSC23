# Notebook README

## Overview
This folder contains all notebooks needed to run the model. The following is a detailed explanation of all the notebooks, their functionalities and how to use them. They are designed to be run in series, however if the necessary preprocessing steps ahve already been performed then you can only run 04_run_training.

## 01_preprocess_waves.ipynb:
Preprocesses waveform data to uniform length and saves the data locally. Use the overlap parameter to determine how much of the data overlaps. For example a 9 second long sound file is covered completely by 2 smaller 5 second sound files with an overlap of 1 second.

## 02_classweights.ipynb:
Calculates different types of class weights. Uses metadata from the "data"-folder as an input and saves class weights in an *.npy file in the "class_weitghts"-folder.

## 03_scan_lr.ipynb:
Trains the given model multiple times with different learning rates for a small number of epochs to figure out the most suitable leartning rate.


## 04_run_training.ipynb:
The main notebook to perform finetuning of a pretrained model or to train a model from scratch. All hyperparameters of the model can be set at the start of the notebook. After the model finished training, all relevant metrics, predictions and logging events will be calculated and saved to storage.


## 05_scan_parameters.ipynb:
Notebook that loops over multiple training parameters to better estimate the best hyperparameter settings.
