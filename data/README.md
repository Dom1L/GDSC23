# Capgemini GDSC 2023 README (Data-Folder)

## Overview 
This folder contains the whole data that is used for the training, evaluation and testing of the selected ML model as well as a metadata csv file. For further details look at the **Data** section below.

## Model Files
This is the expected structure for accessing the given data within the notebooks. Paths with no file ending are folders.

~~~
data/
  train                  Directory containing train data.
  val                    Directory containing validation data.
  test                   Directory containing test data.
  production_data/       Directory containing customized data.
    metadata.csv         Metadata of the customized data.
  metadata.csv           Metadata of the training and validation data.
~~~

## Data
This directory contains the whole data in form of waveform files as well as a metadata csv file, that contains the exact filename, path and label for the training and validation data. Crucially, all data is supposed to have the same sampling frequency, but can vary in length.

In production_data 


| file_name | unique_file | path | species | label | subset | sample_rate | num_frames | lenght |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Header | Title | xxx | xxx | xxx | xxx | xxx | xxx | xxx | 
| Paragraph | Text | xxx | xxx | xxx | xxx | xxx | xxx | xxx | 

## Additional notes
