# Capgemini Global Data Science Challenge 2023 Readme

## Overview
This repository contains code to create, train and evaluate ML models for classifying sound data. The specific task was to classify insect sounds into 66 different classes. This README file provides an overview of the model, its functionality and how to use it. Refer to README files located in the sub-directories for in-depth explanation.  

## Model Description
The ML model is built using PyTorch Lightning and utilizes pre-trained models as part of its evaluation. It is based on the spectrogram approach of audio calssification i.e. transforming raw data into Mel spectrograms and then solving the resulting image classification problem.

## Dependencies
External python libraries, frameworks, or packages that are required to run the model successfully. See also requirements.txt.

- numpy
- tqdm
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch
- lightning
- lightning-bolts
- timm
- tensorboard
- torch-audiomentations
- colorednoise

## Usage
1. Install the necessary dependencies specified in the "Dependencies" section.
2. Clone this repository.
3. Make sure that the data folder is set up correctly.
4. Navigate to the notebooks directory and read the Readme.
5. Decide which hyperparameters to use, whether to use a pretrained model, etc.
6. Run the necessary notebooks including 04_run_training.

## Training the Model
Decide if you want to use a pretrained model, or train the model from scratch. This and other hyperparameters can be set in the 04_run_training notebook.

## Making Predictions or Classifications
Running the 04_run_training notebook creates sub-folders for each training run, containing model checkpoints, hyperparamter savefiles, events.out.tfevents for logging, two prediction csv and other useful files. Refer to the notebooks README.   

## Model Evaluation
Tensorboard is a supported logger for Pytorch Lightning and the saved events.out.tfevents files can be analyzed in tensorboard. Refer to [tensorboard](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.) documentation. 

## Data
This Model is expecting 
Describe the data used to train and evaluate the model. Include information about the dataset's source, format, and any preprocessing steps performed on the data.

## Model Files
This is the expected structure of the model. Paths with no file ending are folders.

~~~
class_weights:                       Directory containing different class weights, based on various metrics.
data:                                Directory containing train, validation and test data.
notebooks/
  01_preprocess_waves.ipynb:         Notebook for preprocessing sound data to uniform length. 
  02_classweights.ipynb:             Notebook that calculates various class weights. 
  03_scan_lr.ipynb:                  Notebook that trains the model multiple times with different learning rates.
  04_run_training.ipynb:             Notebook that trains and evaluates the data. 
  05_scan_parameters.ipynb:  
shell_scripts:                       Directory that contains 2 shell scripts, to help reduce costs on AWS.
src/
  custom/  
    __inti__.py:    
    data.py:    
    eval.py:
    net.py:
    trainer.py
    utils.py
  baseline_ast_train.py 
  config.py
  eda_utils.py
  gdsc_eval.py 
  gdsc_utils.py 
  preprocessing.py  
requirements.txt
~~~

## Additional Notes
Include any additional notes or considerations relevant to the model, such as limitations, known issues, or future improvements.

## Contributors
List of people, that contributed in creating this model:
- Raffaela Heily
- Dominik Lemm
- Lukas Kementinger
- Lucas Unterberger

Many thanks to the brilliant **I&D Germany** team that made all of this possible:
- Daniel Kühlwein
- ...

A special thanks to the team of **Naturalis Biodiversity Center** that provided the data, the idea and the path towards a more sustainable and biodiverse future:
- Max Schöttler
- ....

Additionally we want to thank **AWS**, for providing us with their state of the art cloud infrastructure and enough resources, to make this challenge a reality

## License
Specify the license under which the model is released. If it is an open-source project, provide a link to the license file.
