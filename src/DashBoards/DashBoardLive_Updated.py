# only with live tab
import os

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from lime import lime_tabular

import time  # to simulate a real time data, time loop
import plotly.express as px  # interactive charts

# data preprocessing
def load_and_label_data(base_path, label, max_files=None):
    combined_100KHzdata = []
    combined_2000KHzdata = []

    file_counter = 0

    for timestamp_folder in os.listdir(base_path):
        if max_files and file_counter >= max_files:
            break

        timestamp_folder_path = os.path.join(base_path, timestamp_folder, "raw")
        try:
            timestamp = timestamp_folder.split('_')[0]
            # timestamp = pd.to_datetime(timestamp, format='%Y.%m.%d_%H.%M.%S')
            timestamp = pd.to_datetime(timestamp, format='%Y.%m.%d')
        except ValueError:
            continue

        # Process 2000KHz data
        df_2000KHz = pd.read_parquet(os.path.join(timestamp_folder_path, "Sampling2000KHz_AEKi-0.parquet"))
        df_2000KHz_grouped = df_2000KHz.groupby(df_2000KHz.index // 10000).mean().reset_index(drop=True)
        df_2000KHz_grouped['timestamp'] = timestamp 
        df_2000KHz_grouped['label'] = label

        # Process 100KHz data
        df_100KHz = pd.read_parquet(os.path.join(timestamp_folder_path,
                                                 "Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet"))
        df_100KHz_grouped = df_100KHz.groupby(df_100KHz.index // 10000).mean().reset_index(drop=True)
        df_100KHz_grouped['timestamp'] = timestamp 
        df_100KHz_grouped['label'] = label

        combined_100KHzdata.append(df_100KHz_grouped)
        combined_2000KHzdata.append(df_2000KHz_grouped)

        file_counter += 1

    final_combined_100KHzdata = pd.concat(combined_100KHzdata, ignore_index=True)
    final_combined_2000KHzdata = pd.concat(combined_2000KHzdata, ignore_index=True)

    return final_combined_100KHzdata, final_combined_2000KHzdata


def preprocess_data(ok_data_path, nok_data_path):
    ok_100KHzdata, ok_2000KHzdata = load_and_label_data(ok_data_path, label=0)
    nok_100KHzdata, nok_2000KHzdata = load_and_label_data(nok_data_path, label=1)

    all_100KHzdata = pd.concat([ok_100KHzdata, nok_100KHzdata], ignore_index=True)
    all_2000KHzdata = pd.concat([ok_2000KHzdata, nok_2000KHzdata], ignore_index=True)

    return all_100KHzdata, all_2000KHzdata


def combine_and_interpolate_data(data_100KHz, data_2000KHz):
    # Merge on timestamp
    combined_data = pd.merge_asof(data_100KHz.sort_values('timestamp'),
                                  data_2000KHz.sort_values('timestamp'),
                                  on='timestamp',
                                  by='label',
                                  direction='nearest')

    # Interpolate to fill missing values
    combined_data = combined_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    return combined_data


def normalize_data(combined_data):
    features = combined_data.drop(columns=['timestamp', 'label'])
    timestamps = combined_data['timestamp']
    labels = combined_data['label']

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    normalized_data = pd.DataFrame(normalized_features, columns=features.columns)
    normalized_data.insert(0, 'timestamp', timestamps)
    normalized_data['label'] = labels.values

    return normalized_data


# funtion for model evaluation
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_classes = (y_pred > 0.5).astype("int32")

    # Calculate accuracy precision, recall, and F1 score
    accuracy = accuracy_score(y, y_pred_classes)
    precision = precision_score(y, y_pred_classes)
    recall = recall_score(y, y_pred_classes)
    f1 = f1_score(y, y_pred_classes)

    # Print the classification report
    print(classification_report(y, y_pred_classes))

    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return accuracy, precision, recall, f1


# Define paths to data, please use your own path
ok_data_path = '/Users/ritikahiremath/Downloads/Data/OK_Measurements'
nok_data_path = '/Users/ritikahiremath/Downloads/Data/NOK_Measurements'

# Preprocess data
all_100KHzdata, all_2000KHzdata = preprocess_data(ok_data_path, nok_data_path)

# Combine and interpolate data
combined_data = combine_and_interpolate_data(all_100KHzdata, all_2000KHzdata)

# Normalize data
# normalized_data = normalize_data(combined_data)

# Shuffle the combined data
# normalized_data = shuffle(normalized_data, random_state=42)

# normalized_data.head()

# Model training

# split the data into train and test data
# X = normalized_data.iloc[:, 1:-1]
# y = normalized_data['label']

combined_data = combined_data.rename(columns={'Irms_Grinding_rate100000_clipping0_batch0': 'Irms_Grinding_rate',
                                              'Grinding spindle current L1_rate100000_clipping0_batch0': 'Spindle L1_rate',
                                              'Grinding spindle current L2_rate100000_clipping0_batch0': 'Spindle L2_rate',
                                              'Grinding spindle current L3_rate100000_clipping0_batch0': 'Spindle L3_rate',
                                              'AEKi_rate2000000_clipping0_batch0': 'AEKi_rate2'})


feature_names = ['Irms_Grinding_rate',
                 'Spindle L1_rate',
                 'Spindle L2_rate',
                 'Spindle L3_rate',
                 'AEKi_rate2']

target = ['label']
target_names = ['0', '1']

X = combined_data[feature_names]
y = combined_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#train the model
xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(X_train, y_train)

y_test_preds = xgb_classifier.predict(X_test)

# Evaluate the model on training data
train_accuracy, train_precision, train_recall, train_f1 = evaluate_model(xgb_classifier, X_train, y_train)
print(
    f"Training Accuracy: {train_accuracy * 100:.2f}%, Training Precision: {train_precision:.2f}, Training Recall: {train_recall:.2f}, Training F1 Score: {train_f1:.2f}")

# Evaluate the model on test data
test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(xgb_classifier, X_test, y_test)
print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%, Test Precision: {test_precision:.2f}, Test Recall: {test_recall:.2f}, Test F1 Score: {test_f1:.2f}")

# Create a results table
results = {
    "Dataset": ["Training", "Test"],
    "Accuracy": [train_accuracy, test_accuracy],
    "Precision": [train_precision, test_precision],
    "Recall": [train_recall, test_recall],
    "F1 Score": [train_f1, test_f1]
}

results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df)
st.set_page_config( page_title="Real-Time Data Science Dashboard", page_icon="✅", layout="wide")
# top-level filters
label_filter = st.selectbox("Select the date", pd.unique(combined_data['timestamp']))

# dataframe filter
combined_data = combined_data[combined_data['timestamp'] == label_filter]

feature_names = ['Irms_Grinding_rate',
                 'Spindle L1_rate',
                 'Spindle L2_rate',
                 'Spindle L3_rate',
                 'AEKi_rate2']
feature_new_values = ['Irms_new',
                      'L1_rate_new',
                      'L2_rate_new',
                      'L3_rate_new',
                      'L3_rate_new',
                      'AEKi_rate2_new']

st.title("Grinding Status :red[Prediction] ")
st.markdown("Predict Grinding Status Using Acoustic Emission Values")

# near real-time / live feed simulation
st.header("Grinding Live")
tab0, tab1, tab2 = st.tabs(
    ["Live Detection ", "Data", "Model Performance"])
with tab0: 
    # creating a single-element container
    placeholder = st.empty()

    for seconds in range(200):
        # to better display the parameter value, multiple them with 100-105, otherwise they are too small to be shown in the dashboard
        combined_data['Irms_new'] = combined_data['Irms_Grinding_rate'] * np.random.choice(range(100, 105))
        combined_data['L1_rate_new'] = combined_data['Spindle L1_rate'] * np.random.choice(range(100, 105))
        combined_data['L2_rate_new'] = combined_data['Spindle L2_rate'] * np.random.choice(range(100, 105))
        combined_data['L3_rate_new'] = combined_data['Spindle L3_rate'] * np.random.choice(range(100, 105))
        combined_data['AEKi_rate2_new'] = combined_data['AEKi_rate2'] * np.random.choice(range(100, 105))

        # create to be displayed parameters
        avg_Irms = np.mean(combined_data['Irms_new'])

        avg_L1 = np.mean(combined_data['L1_rate_new'])

        avg_L2 = np.mean(combined_data['L2_rate_new'])

        avg_L3 = np.mean(combined_data['L3_rate_new'])

        avg_AEKi = np.mean(combined_data['AEKi_rate2_new'])

        with placeholder.container():
            # create five columns
            sensor1, sensor2, sensor3, sensor4, sensor5 = st.columns(5)

            # fill in those five columns with respective metrics
            sensor1.metric(label="Irms", value=avg_Irms)  # Indicator of how the metric changed, rendered with an arrow below the metric.
            sensor2.metric(label="L1_rate", value=avg_L1)
            sensor3.metric(label="L2_rate", value=avg_L2)
            sensor4.metric(label="L3_rate", value=avg_L3)
            sensor5.metric(label="AEKi_rate2", value=avg_AEKi)

            # create two columns for classification

            data = [[avg_Irms/100, avg_L1/100, avg_L2/100, avg_L3/100, avg_AEKi/100]]
            df = pd.DataFrame(data, columns=['Irms_Grinding_rate','Spindle L1_rate','Spindle L2_rate','Spindle L3_rate','AEKi_rate2'])
            prediction = xgb_classifier.predict(df.to_numpy())

            predict="Normal"
            if(prediction[0]==1):
                predict="Anomaly"
            st.markdown("prediction:", prediction[0])
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(
                predict), unsafe_allow_html=True)
            st.markdown("### Feature plot - Live ")
            # fig = px.line(combined_data, y=feature_names, title='Feature Values Over Time')
            # st.plotly_chart(fig)
            col1, col2= st.columns(2)
            with col1:
                st.header("Irms")
                st.line_chart(combined_data, y='Irms_Grinding_rate')

            with col2:
                st.header("AEKi_rate2")
                st.line_chart(combined_data, y='AEKi_rate2' )

            col1, col2 , col3 = st.columns(3)
            with col1:
                st.header("Spindle L1_rate")
                st.line_chart(combined_data, y='Spindle L1_rate')

            with col2:
                st.header("Spindle L2_rate")
                st.line_chart(combined_data, y='Spindle L2_rate' )

            with col3:
                st.header("Spindle L3_rate")
                st.line_chart(combined_data, y='Spindle L3_rate')


            st.markdown("### Detailed raw Data View")
            st.dataframe(df)

            # Initialize the chart
            # st.title("Live Data Stream")
            # df = pd.DataFrame(combined_data)
            # df = df.set_index('timestamp')
            # chart = st.line_chart(df)

            # Simulate live data
            # for i in range(100, 200):
            #     new_data = {
            #         'timestamp': timestamps.append(pd.date_range(start=timestamps[-1] + pd.Timedelta(seconds=1), periods=1)),
            #         'Irms_Grinding_rate': np.random.rand(1) * 100,
            #         'Spindle L1_rate': np.random.rand(1) * 100,
            #         'Spindle L2_rate': np.random.rand(1) * 100,
            #         'Spindle L3_rate': np.random.rand(1) * 100,
            #         'AEKi_rate2': np.random.rand(1) * 100
            #     }
            #     new_df = pd.DataFrame(new_data)
            #     new_df = new_df.set_index('timestamp')

            #     df = pd.concat([df, new_df]).tail(100)  # Keep only the last 100 points
            #     chart.add_rows(new_df)
            #     time.sleep(1)
        time.sleep(1)
with tab1:
    st.header("Grinding Data")
    st.write(combined_data)

with tab2:
    st.header("Model Performance")
    st.write(results_df)