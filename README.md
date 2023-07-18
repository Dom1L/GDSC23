# Capgemini Global Data Science Challenge 2023 Readme

## Overview
This repository contains code to create, train and evaluate ML models for classyfing sound data. The specific task was to classify insect sounds into 66 different classes. This README file provides an overview of the model, its functionality and how to use it.  

## Model Description
The ML model is built using TorchAudio and utilizes pre-trained models as part of its evaluation. It is based on the spectrogram approach of audio calssification i.e. transforming raw data into Mel spectrograms and then solve an image classification problem.

## Dependencies
External libraries, frameworks, or packages that are required to run the model successfully. See also requirements.txt.

1. Library Name 1 (Version X.X.X)
1. Library Name 2 (Version X.X.X)
1. Library Name 1 (Version X.X.X)
1. Library Name 2 (Version X.X.X)
1. Library Name 1 (Version X.X.X)
1. Library Name 2 (Version X.X.X)


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
Describe how to use the trained model to make predictions or perform classifications. Include code snippets or instructions for running the model on new data.

## Model Evaluation
If applicable, describe how to evaluate the performance of the model, including metrics used and how to interpret the results.

## Data
Describe the data used to train and evaluate the model. Include information about the dataset's source, format, and any preprocessing steps performed on the data.

## Model Files
This is the expected structure of the model. Paths with no file ending are folders.

class_weights: Directory containing different class weights, based on various metrics.

data: Directory containing train, validation and test data.

notebooks/

  01_preprocess_waves.ipynb: Notebook for preprocessing sound data to uniform length.
  
  02_classweights.ipynb: Notebook that calculates various class weights.
  
  03_scan_lr.ipynb: Notebook that trains the model multiple times with different learning rates for a few epochs.
  
  04_run_training.ipynb: Notebook that trains and evaluates the data.
  
  05_scan_parameters.ipynb: 
  
shell_scripts: Directory that contains 2 shell scripts, to help reduce costs on AWS.

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

## Additional Notes
Include any additional notes or considerations relevant to the model, such as limitations, known issues, or future improvements.

## Contributors
List the names or usernames of the contributors who developed the model.

## License
Specify the license under which the model is released. If it is an open-source project, provide a link to the license file.
