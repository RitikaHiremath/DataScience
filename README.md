# Grinding process monitoring System: Anomaly detection and prediction

## Overview

This project trains an LSTM model for classification tasks by processing and analyzing sensor data from two distinct sampling rates (100KHz and 2000KHz). The labels for the data are NOK (label 1) or OK (label 0). The project involves loading, preprocessing, normalizing, combining, and sequencing the data, after which the processed data is used to train an LSTM model.
1. For accessing the data, Google drive is mounted.
2. There are two directories of data: 
OK_Measurements: Directory containing OK sensor data.
NOK_Measurements: Directory containing NOK sensor data.
3.Pre-processing of data includes Labeling and normalizing the data. Then the two data frames are combined together and sequenced in ascending order of timestamp.
4. Finally, an LSTM MODEL is defined and trained, along with plotting Training and validation metrics.
