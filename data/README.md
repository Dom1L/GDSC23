# Capgemini GDSC 2023 README (Data-Folder)

## Overview 
This folder contains the whole data that is used for the training, evaluation and testing of the selected ML model as well as a metadata csv file. For detailed information look at the **Data** section below.

## Model Files
This is the expected structure for accessing the given data within the notebooks. Paths with no file ending are folders.

~~~
data/
  train                   Directory containing train data.
  val                     Directory containing validation data.
  test                    Directory containing test data.
  production_data/        Directory containing customized data.
    crop-x-s/             Directory with data at a given uniform lenght of x seconds
       train              Directory containing the customized validation data.
       val                Directory containing the customized validation data.
       metadata.csv       Metadata of the customized data.
    ...
  metadata.csv            Metadata of the training and validation data.
~~~

## Data
This directory contains the whole data in form of waveform files as well as a metadata csv file, that contains the exact filename, path and label for the training and validation data. Crucially, all data is supposed to have the same sampling frequency, but can vary in length.
The metadata file 


| file_name | unique_file | path | species | label | subset | sample_rate | num_frames | lenght |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| insect_dae_003.wav | insect_dae_003 | data/val/insect_dae_003.wav | insect_dae | 49 | validation | 44100 | 30870 | 7 |  
| insect_bcd_001_dat1.wav | insect_bcd_001_dat1 | data/val/insect_bcd_001_dat1.wav | insect_bcd | 34 | validation | 44100 | 88200 | 2 | 
| insect_abc_001.wav | insect_abc_001 | data/train/insect_abc_001.wav | insect_abc | 1 | train | 44100 | 4463050 | 10.5 | 
| insect_dae_002.wav | insect_dae_002 | data/val/insect_dae_002.wav | insect_dae | 49 | validation | 44100 | 238140 | 5.4 |
| insect_abc_002_edit.wav | insect_abc_002_edit | data/train/insect_abc_002_edit.wav | insect_abc | 1 | train | 44100 | 337571 | 7.65467 | 
| insect_abb_001.wav | insect_abb_001 | data/train/insect_abb_001.wav | insect_abb | 7 | train | 44100 | 502740 | 11.4 | 


After using the 01_preprocess_waves notebooks a folder with data that contains waveforms at a given uniform lenght is created and placed into the production_data folder. This folder contains a metadata for the created files.
For more detailed information look at the README in the notebooks folder.
Note that the metadata for this customized files has a different sctructure, given as follows:

| file_name | unique_file | path | label | subset |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| insect_dae_003.wav | insect_dae_003 | data/production_data/crop-x-s/val/insect_dae_003_chunk1.wav | 49 | validation | 
| insect_dae_003.wav | insect_dae_003 | data/production_data/crop-x-s/val/insect_dae_003_chunk1.wav | 49 | validation | 
| insect_bcd_001_dat1.wav | insect_bcd_001_dat1 | data/production_data/crop-x-sval/insect_bcd_001_dat1_loop.wav | 34 | validation | 
| insect_bcd_001_dat1.wav | insect_bcd_001_dat1 | data/production_data/crop-x-sval/insect_bcd_001_dat1_padded.wav | 34 | validation | 
| insect_abc_001.wav | insect_abc_001 | data/production_data/crop-x-s/train/insect_abc_001_chunk1.wav | 1 | train |

## Additional notes
