# Capgemini Global Data Science Challenge 2023 README

## Overview
This repository contains code to create, train and evaluate ML models for classifying sound data. The specific task was to classify insect sounds into 66 different classes. This README file provides an overview of the model, its functionality and how to use it. Refer to README files located in the sub-directories for in-depth explanation.  

## Model Description
The ML model is built using PyTorch [Lightning](https://www.pytorchlightning.ai/index.html) and utilizes pre-trained models as part of its evaluation. It is based on the spectrogram approach of audio classification i.e. transforming raw waveform data into Mel spectrograms and then solving the resulting image classification problem. We use the small pretrained [EfficientNetv2](https://github.com/google/automl/tree/master/efficientnetv2) to solve the computer vision problem, which is as the name implies quite efficient to fine-tune and run. It utilizes ~20 million parameters, thus achieves very fast training speed and still provides state of the art performance. Link to the supporting research: [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

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
4. Navigate to the notebooks directory and read the README.
5. Decide which hyperparameters to use, whether to use a pretrained model, etc.
6. Run the necessary notebooks including 04_run_training.

## Pretrained vs Trained Model
Decide if you want to fine-tune a pretrained model, or train a model from scratch. This and other hyperparameters can be set in the 04_run_training notebook.

## Making Classifications
Running the 04_run_training notebook creates sub-folders for each training run, containing model checkpoints, hyperparamter savefiles, events.out.tfevents for logging, two prediction csv and other useful files. Refer to the notebooks README.   

## Data
This Model is expecting waveform files in the data folder together with a metadata csv file, that contains the exact filename, path and label for training/validation data and filename, path for test data. Crucially, all data is supposed to have the same sampling frequency, but can vary in length.
We split every audio file into smaller windows of uniform length. We made this decision since we noticed a big deviation of audio length between our training data files. So for example a 27 second long file could be split into 6 smaller files, each 5 seconds long, with the last file padded with zeros.

| Syntax | Description |
| ----------- | ----------- |
| Header | Title |
| Paragraph | Text |

## Model Evaluation
We evaluate a single audio file by calculating the prediction of each uniform subfile, then averaging all predictions for that singular file. Finally, we calculate the f1-score, among other metrics, based on all predictions. In addition we return the confusion matrix and use Tensorboard logging, which is a supported logger for Pytorch Lightning, during our training runs. The saved events.out.tfevents files can be analyzed in tensorboard, to monitor all desirable metrics during the training run. For further details, refer to [tensorboard](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.) documentation. 

## Model Files
This is the expected structure to run the model successfully. Paths with no file ending are folders.

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
- Lukas Kementinger
- Dominik Lemm
- Lucas Unterberger

Special thanks to the team of **Naturalis Biodiversity Center** that provided the data, the idea and the path towards a more sustainable and biodiverse future:
- Dr. Elaine van Ommen Kloeke 
- Max Schöttler
- Dr. Dan Stowell
- Marius Faiß

Many thanks to our Capgemini Sponsors and organizational team that made all of this possible:
- Niraj Parihar
- Susanna Ostberg
- TP Deo
- Anne-Laure Thiuellent
- Andris Roling
- Dr. Daniel Kühlwein
- Kanwalmeet Singh Kochar
- Kristin O'Herlihy
- Marc Niedermeier
- Mateusz Gryz
- Nikolai Babic
- Sebastian Sell
- Sophie (Nien-chun) Yin
- Steffen Klempau
- Surobhi Deb
- Timo Abele
- Tomasz Czerniawski

Additionally we want to thank **AWS**, for providing us with their state of the art cloud infrastructure and enough resources, to make this challenge a reality.

## License
Specify the license under which the model is released. If it is an open-source project, provide a link to the license file.
