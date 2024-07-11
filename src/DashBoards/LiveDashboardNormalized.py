import os
import random
import time

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


# data preprocessing
def load_and_label_data(base_path, label, max_files=None):
    combined_100KHzdata = []
    combined_2000KHzdata = []

    file_counter = 0

    for timestamp_folder in os.listdir(base_path):
        if max_files and file_counter >= max_files:
            break

        timestamp_folder_path = os.path.join(base_path, timestamp_folder, "raw")
        timestamp = timestamp_folder.split('_')[0] + '_' + timestamp_folder.split('_')[1]
        timestamp = pd.to_datetime(timestamp, format='%Y.%m.%d_%H.%M.%S')

        # Process 2000KHz data
        df_2000KHz = pd.read_parquet(os.path.join(timestamp_folder_path, "Sampling2000KHz_AEKi-0.parquet"))
        df_2000KHz_grouped = df_2000KHz.groupby(df_2000KHz.index // 10000).mean().reset_index(drop=True)
        df_2000KHz_grouped['timestamp'] = timestamp + pd.to_timedelta(df_2000KHz_grouped.index, unit='ms')
        df_2000KHz_grouped['label'] = label

        # Process 100KHz data
        df_100KHz = pd.read_parquet(os.path.join(timestamp_folder_path,
                                                 "Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet"))
        df_100KHz_grouped = df_100KHz.groupby(df_100KHz.index // 10000).mean().reset_index(drop=True)
        df_100KHz_grouped['timestamp'] = timestamp + pd.to_timedelta(df_100KHz_grouped.index, unit='ms')
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


def col_rename(dataset):
    dataset = dataset.rename(columns={'Irms_Grinding_rate100000_clipping0_batch0': 'Irms_rate',
                                      'Grinding spindle current L1_rate100000_clipping0_batch0': 'L1_rate',
                                      'Grinding spindle current L2_rate100000_clipping0_batch0': 'L2_rate',
                                      'Grinding spindle current L3_rate100000_clipping0_batch0': 'L3_rate',
                                      'AEKi_rate2000000_clipping0_batch0': 'AEKi_rate'})
    return dataset


# Define paths to data, please use your own path
ok_data_path = ''
nok_data_path = ''

# Preprocess data
all_100KHzdata, all_2000KHzdata = preprocess_data(ok_data_path, nok_data_path)

# Combine and interpolate data
combined_data = combine_and_interpolate_data(all_100KHzdata, all_2000KHzdata)

# Normalize data
normalized_data = normalize_data(combined_data)

# Shuffle the combined data
normalized_data = shuffle(normalized_data, random_state=42)

# normalized_data.head()

# Model training

# split the data into train and test data
# X = normalized_data.iloc[:, 1:-1]
# y = normalized_data['label']

normalized_data = col_rename(normalized_data)

combined_data = col_rename(combined_data)

combined_data = combined_data.reindex(['timestamp', 'Irms_rate', 'L1_rate', 'L2_rate', 'L3_rate',
                                       'AEKi_rate', 'label'], axis=1)

feature_names = ['Irms_rate', 'L1_rate', 'L2_rate', 'L3_rate', 'AEKi_rate']
target = ['label']
target_names = ['0', '1']

X = normalized_data[feature_names]
y = normalized_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#train the model
xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(X_train, y_train)

y_test_preds = xgb_classifier.predict(X_test)

# Evaluate the model on training data
train_accuracy, train_precision, train_recall, train_f1 = evaluate_model(xgb_classifier, X_train, y_train)

# Evaluate the model on test data
test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(xgb_classifier, X_test, y_test)

# Create a results table
results = {
    "Dataset": ["Training", "Test"],
    "Accuracy": [train_accuracy, test_accuracy],
    "Precision": [train_precision, test_precision],
    "Recall": [train_recall, test_recall],
    "F1 Score": [train_f1, test_f1]
}

results_df = pd.DataFrame(results)
# print("\nResults Summary:")
# print(results_df)

st.title("Grinding Status :red[Prediction]")
st.markdown("Predict Grinding Status Using Acoustic Emission Values")

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Live Detection", "Dataset", "Model Performance", "Global Performance",
     "Local Prediction"])

with tab0:
    # creating a single-element container
    placeholder = st.empty()

    previous = [0] * 5

    # top-level filters
    label_filter = st.selectbox("Select the date", pd.unique(normalized_data['timestamp'].dt.date))

    # dataframe filter
    normalized_data = normalized_data[normalized_data['timestamp'].dt.date == label_filter]

    # near real-time / live feed simulation
    # simulated live detection: default by 60 seconds, can also use while loop to run endlessly
    for seconds in range(60):
        # create to be displayed parameters
        Irms = random.uniform(normalized_data['Irms_rate'].min(), normalized_data['Irms_rate'].max())
        L1 = random.uniform(normalized_data['L1_rate'].min(), normalized_data['L1_rate'].max())
        L2 = random.uniform(normalized_data['L2_rate'].min(), normalized_data['L2_rate'].max())
        L3 = random.uniform(normalized_data['L3_rate'].min(), normalized_data['L3_rate'].max())
        AEKi = random.uniform(normalized_data['AEKi_rate'].min(), normalized_data['AEKi_rate'].max())

        with placeholder.container():
            # create five columns
            sensor1, sensor2, sensor3, sensor4, sensor5 = st.columns(5)

            # fill in those five columns with respective metrics
            # to better display the parameter value, multiple them with 100-105,
            # otherwise they are too small to be shown in the dashboard
            sensor1.metric(label="Irms", value=Irms, delta=Irms - previous[0])
            sensor2.metric(label="L1_rate", value=L1, delta=L1 - previous[1])
            sensor3.metric(label="L2_rate", value=L2, delta=L2 - previous[2])
            sensor4.metric(label="L3_rate", value=L3, delta=L3 - previous[3])
            sensor5.metric(label="AEKi_rate2", value=AEKi, delta=AEKi - previous[4])

            # update "previous" list
            previous = [Irms, L1, L2, L3, AEKi]

            # create two columns for classification

            data = [[Irms, L1, L2, L3, AEKi]]
            df = pd.DataFrame(data,
                              columns=['Irms_rate', 'L1_rate', 'L2_rate', 'L3_rate',
                                       'AEKi_rate'])
            prediction = xgb_classifier.predict(df.to_numpy())

            predict = "Normal"
            if prediction[0] == 1:
                predict = "Anomaly"
            st.markdown("prediction:", prediction[0])
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(
                predict), unsafe_allow_html=True)
            st.markdown("### Feature plot - Live ")

            # fig = px.line(combined_data, y=feature_names, title='Feature Values Over Time')
            # st.plotly_chart(fig)
            col1, col2 = st.columns(2)
            # Shuffle the normalized data for plots
            normalized_data = shuffle(normalized_data, random_state=42)
            with col1:
                st.header("Irms")
                st.line_chart(normalized_data, y='Irms_rate')

            with col2:
                st.header("AEKi_rate2")
                st.line_chart(normalized_data, y='AEKi_rate')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.header("Spindle L1_rate")
                st.line_chart(normalized_data, y='L1_rate')

            with col2:
                st.header("Spindle L2_rate")
                st.line_chart(normalized_data, y='L2_rate')

            with col3:
                st.header("Spindle L3_rate")
                st.line_chart(normalized_data, y='L3_rate')

        time.sleep(1)

with tab1:
    st.header("Grinding Dataset")
    st.write(combined_data)

with tab2:
    st.header("Model Performance")
    st.write(results_df)

with tab3:
    st.header("Confusion Matrix | Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        conf_mat_fig = plt.figure(figsize=(6, 6))
        ax1 = conf_mat_fig.add_subplot(111)
        skplt.metrics.plot_confusion_matrix(y_test, y_test_preds, ax=ax1, normalize=True)
        st.pyplot(conf_mat_fig, use_container_width=True)

    with col2:
        feat_imp_fig = plt.figure(figsize=(6, 6))
        ax1 = feat_imp_fig.add_subplot(111)
        skplt.estimators.plot_feature_importances(xgb_classifier, feature_names=feature_names, ax=ax1,
                                                  x_tick_rotation=90)
        st.pyplot(feat_imp_fig, use_container_width=True)

    #st.divider()
    #st.header("Classification Report")
    #st.code(classification_report(y_test, y_test_preds))

with tab4:
    sliders = []
    col1, col2 = st.columns(2)
    with col1:
        for ingredient in feature_names:
            ing_slider = st.slider(label=ingredient, min_value=float(normalized_data[ingredient].min()),
                                   max_value=float(normalized_data[ingredient].max()))
            sliders.append(ing_slider)

    with col2:
        col1, col2 = st.columns(2, gap="medium")

        prediction = xgb_classifier.predict([sliders])
        with col1:
            predict = "Normal"
            if target_names[prediction[0]] == 1:
                predict = "Anomaly"
            st.markdown("prediction:", prediction[0])
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(
                predict), unsafe_allow_html=True)

        probs = xgb_classifier.predict_proba([sliders])
        probability = probs[0][prediction[0]]

        with col2:
            st.metric(label="Model Confidence", value="{:.2f} %".format(probability * 100),
                      delta="{:.2f} %".format((probability - 0.5) * 100))

        explainer = lime_tabular.LimeTabularExplainer(X_train.to_numpy(),
                                                      mode="classification",
                                                      class_names=target_names,
                                                      feature_names=feature_names)
        explanation = explainer.explain_instance(np.array(sliders), xgb_classifier.predict_proba,
                                                 num_features=len(feature_names), top_labels=2)
        interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
        st.pyplot(interpretation_fig, use_container_width=True)
