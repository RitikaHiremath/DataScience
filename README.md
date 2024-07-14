# Grinding process monitoring System: Anomaly detection and prediction

## AEGaurd Model

In the first part, this project trains an LSTM model for classification tasks by processing and analyzing grinding machine sensor data from two distinct sampling rates (100KHz and 2000KHz). The labels for the data are NOK (label 1) or OK (label 0). The project involves loading, preprocessing, normalizing, combining, and sequencing the data, after which the processed data is used to train an LSTM model called AEGaurd.
1. For accessing the data, Google drive is mounted.
2. There are two directories of data: 
OK_Measurements: Directory containing OK sensor data.
NOK_Measurements: Directory containing NOK sensor data.
3. Pre-processing of data includes Labeling and normalizing the data. Then the two data frames are combined together and sequenced in ascending order of timestamp.
4. Finally, an LSTM MODEL is defined and trained, along with plotting Training and validation metrics.

## Simulated Live Dash Board

In the second part, a simulated live dashboard was build with the open source tool Streamlit. The implememted dashboard shows how our model would do near-real-time grinding anomaly detection tasks. The dashboard also shows the most important model performances measured with metrics that will be displayed in our dashboard. In addition, we also implemented a local prediction tab to show how a change of a single feature would affect our model prediction results. The current Dashboard has 5 tabs:
Tab 1: conducts the live detection task, where the simulated data will be feed into our AEGaurd model and the predicted results will be shown and updated in every second by default; The value change of the sensor data will also be displayed; The occurred anomalies will also be captured and stored in a table with capture time and sensor data; The overall change of sensor data will also be visualized with plots in this tab; A filter is set to help change the date for feeding corresponding sensor data for live simulation.
Tab 2: displays our processed raw sensor data without normalization, where thetimestamp and label (indicating machine working status) are also included.
Tab 3: shows the model performance in the training and testing dataset with metrics of accuracy, precision, recall, and F1 score.
Tab 4: includes the confusion matrix for measuring model performance. In addition to that, we also added a feature importance diagram to show how much every feature contributes to building our model.
Tab 5: puts the normalized max and min values of each sensor in sliders, where we can change the value of a sensor for anomaly detection tasks at a time. In this way, we can see how would a value change of a single feature can affect the overall prediction result (which corresponds to the partial dependence analysis from the general explainable AI method). In addition to that, we show the prediction result, our model confidence, and its (confidence) change. By this means, our "black-box" model AEGuard can be more transparent and interpretable, which ought to be more trustworthy.

## Required Python Libraries' Version for runing Dash Board (especially windows system) 
For Windows system, if the newest libraries' version not fully support oura application, please consider using the following libraries' version to avoid possible problems:
matplotlib: 3.9.0
numpy: 1.26.4
scikit-learn: 1.5.0
scipy: 1.11.4
