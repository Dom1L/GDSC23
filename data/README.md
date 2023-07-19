# Capgemini GDSC 2023 README (Data-Folder)

## Overview 
This folder contains the whole data that is used for the training, evaluation and testing of the selected ML model as well as a metadata csv file. For detailed information look at the **Data** section below.

## Model Files
This is the expected structure for accessing the given data within the notebooks. Paths with no file ending are folders.

~~~
data/
  train                  Directory containing train data.
  val                    Directory containing validation data.
  test                   Directory containing test data.
  production_data/       Directory containing customized data.
    crop-x-s             Directory with data at a given uniforme lenght
       metadata.csv      Metadata of the customized data.
    ...
  metadata.csv           Metadata of the training and validation data.
~~~

## Data
This directory contains the whole data in form of waveform files as well as a metadata csv file, that contains the exact filename, path and label for the training and validation data. Crucially, all data is supposed to have the same sampling frequency, but can vary in length.
The metadata file 


| file_name | unique_file | path | species | label | subset | sample_rate | num_frames | lenght |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| insect_abc_001.wav | insect_abc_001 |data/train/insect_abc_001.wav | insect_abc | 1 | train | 44100 | 4586400 | 104 | 
| insect_abc_002_edit.wav | insect_abc_002_edit | data/train/insect_abc_002_edit.wav | insect_abc | 1 | train | 44100 | 337571 | 7.65467 | 
| insect_abb_001.wav | insect_abb_001 | data/train/insect_abb_001.wav | insect_abb | 7 | train | 44100 | 220500 | 11.4 | 
| insect_bcd_001_dat1.wav | insect_bcd_001_dat1 | data/val/insect_bcd_001_dat1.wav | insect_bcd | 34 | validation | 44100 | 88200 | 2 | 
| insect_dae_003.wav | insect_dae_003 | data/val/insect_dae_003.wav | insect_dae | 49 | validation | 44100 | 238140 | 5.4 | 

After using the 01_preprocess_waves notebooks a 

## Additional notes
