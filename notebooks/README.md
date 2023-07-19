# Notebook README

## Overview
This folder contains all notebooks needed to run the model. The following is a detailed explanation of all the notebooks, their functionalities and how to use them. They are designed to be run in series, however if you already performed the necessary preprocessing steps then you can also only run 04_run_training.
## 01_preprocess_waves.ipynb:
Preprocesses sound data to uniform length.


## 02_classweights.ipynb:
Calculates different types of class weights. 
Uses metadata from the "data"-folder as an input and saves class weights in an *.npy file in the "class_weitghts"-folder.


## 03_scan_lr.ipynb:
Trains the given model multiple times with different learning rates to figure out the most suitable leartning rate.


## 04_run_training.ipynb:
Trains and evaluates the data.


## 05_scan_parameters.ipynb:
TBD
