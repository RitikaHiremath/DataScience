# DashBoard for our AEGuard model
import os
import sys
import random
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from pre_processing import preprocess_data, combine_and_interpolate_data, normalize_data, create_labeled_sequences
from model import evaluate_model
from loss_function import CustomLoss


def col_rename(dataset):
    dataset = dataset.rename(columns={'Irms_Grinding_rate100000_clipping0_batch0': 'Irms_rate',
                                      'Grinding spindle current L1_rate100000_clipping0_batch0': 'L1_rate',
                                      'Grinding spindle current L2_rate100000_clipping0_batch0': 'L2_rate',
                                      'Grinding spindle current L3_rate100000_clipping0_batch0': 'L3_rate',
                                      'AEKi_rate2000000_clipping0_batch0': 'AEKi_rate'})
    return dataset


# Load the trained Keras model
model = load_model(os.path.join(src_dir, '..', 'AEGuard.keras'), custom_objects={'CustomLoss': CustomLoss})

# Define paths to data
current_directory = os.getcwd()
ok_data_path = os.path.abspath(os.path.join(src_dir, '..', '..', 'Data', 'OK_Measurements'))
nok_data_path = os.path.abspath(os.path.join(src_dir, '..', '..', 'Data', 'NOK_Measurements'))

# Alternative: please use your own data path your data
# ok_data_path = ''
# nok_data_path = ''

# Preprocess data
all_100KHzdata, all_2000KHzdata = preprocess_data(ok_data_path, nok_data_path)

# Combine and interpolate data
combined_data = combine_and_interpolate_data(all_100KHzdata, all_2000KHzdata)

# Normalize data
normalized_data = normalize_data(combined_data)

# Shuffle the combined data
normalized_data = shuffle(normalized_data, random_state=42)

# Rename columns for consistency
normalized_data = col_rename(normalized_data)
combined_data = col_rename(combined_data)

combined_data = combined_data.reindex(['timestamp', 'Irms_rate', 'L1_rate', 'L2_rate', 'L3_rate',
                                       'AEKi_rate', 'label'], axis=1)

feature_names = ['Irms_rate', 'L1_rate', 'L2_rate', 'L3_rate', 'AEKi_rate']
target = ['label']
target_names = ['0', '1']

# get current data for anomalies detection table
curr_time = time.localtime()
curr_date = time.strftime("%d.%m.%Y_%H:%M:%S", curr_time)
# empty table for storing detected anomalies
anomalies = pd.DataFrame(columns=['Detected_time', 'Irms_rate', 'L1_rate', 'L2_rate', 'L3_rate',
                                  'AEKi_rate'])

# Create labeled sequences
sequence_length = 10
X, y = create_labeled_sequences(normalized_data, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_loss, train_accuracy, train_precision, train_recall, train_f1 = evaluate_model(model, X_train, y_train)
test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, X_test, y_test)

# Create a results table
results = {
    "Dataset": ["Training", "Test"],
    "Accuracy": [train_accuracy, test_accuracy],
    "Precision": [train_precision, test_precision],
    "Recall": [train_recall, test_recall],
    "F1 Score": [train_f1, test_f1]
}

results_df = pd.DataFrame(results)

st.title("Grinding Status :red[Detection]")
st.markdown("Detect Anomaly in Grinding System Using Acoustic Emission Values")

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
    normalized_data_filtered = normalized_data[normalized_data['timestamp'].dt.date == label_filter]

    # near real-time / live feed simulation
    buffer = []

    for seconds in range(120):
        Irms = random.uniform(normalized_data_filtered['Irms_rate'].min(), normalized_data_filtered['Irms_rate'].max())
        L1 = random.uniform(normalized_data_filtered['L1_rate'].min(), normalized_data_filtered['L1_rate'].max())
        L2 = random.uniform(normalized_data_filtered['L2_rate'].min(), normalized_data_filtered['L2_rate'].max())
        L3 = random.uniform(normalized_data_filtered['L3_rate'].min(), normalized_data_filtered['L3_rate'].max())
        AEKi = random.uniform(normalized_data_filtered['AEKi_rate'].min(), normalized_data_filtered['AEKi_rate'].max())

        # Add the new data point to the buffer
        buffer.append([Irms, L1, L2, L3, AEKi])

        with placeholder.container():
            sensor1, sensor2, sensor3, sensor4, sensor5 = st.columns(5)

            sensor1.metric(label="Irms", value=Irms, delta=Irms - previous[0])
            sensor2.metric(label="L1_rate", value=L1, delta=L1 - previous[1])
            sensor3.metric(label="L2_rate", value=L2, delta=L2 - previous[2])
            sensor4.metric(label="L3_rate", value=L3, delta=L3 - previous[3])
            sensor5.metric(label="AEKi_rate2", value=AEKi, delta=AEKi - previous[4])

            previous = [Irms, L1, L2, L3, AEKi]

            buffer.append([Irms, L1, L2, L3, AEKi])

            if len(buffer) >= sequence_length:
                # When buffer is full, use the last `sequence_length` points for prediction
                data = np.array(buffer[-sequence_length:])
                data = data.reshape((1, sequence_length,
                                     len(feature_names)))  # Reshape the data to match the model's expected input shape
                prediction = model.predict(data)
                prediction_class = (prediction > 0.5).astype("int32")

                predict = "Normal"
                if prediction_class[0][0] == 1:
                    predict = "Anomaly"
                    new_anomalies = {'Detected_time': curr_date, 'Irms_rate': Irms, 'L1_rate': L1, 'L2_rate': L2,
                                     'L3_rate': L3, 'AEKi_rate': AEKi}
                    anomalies.loc[len(anomalies)] = new_anomalies
                st.markdown("Prediction:", prediction_class[0][0])
                st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(
                    predict), unsafe_allow_html=True)

            st.header("Detected :red[Anomalies]")
            st.write(anomalies)

            st.markdown("### Feature plot - Live ")

            col1, col2 = st.columns(2)
            normalized_data_filtered = shuffle(normalized_data_filtered, random_state=42)
            with col1:
                st.header("Irms")
                st.line_chart(normalized_data_filtered['Irms_rate'])

            with col2:
                st.header("AEKi_rate2")
                st.line_chart(normalized_data_filtered['AEKi_rate'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.header("Spindle L1_rate")
                st.line_chart(normalized_data_filtered['L1_rate'])

            with col2:
                st.header("Spindle L2_rate")
                st.line_chart(normalized_data_filtered['L2_rate'])

            with col3:
                st.header("Spindle L3_rate")
                st.line_chart(normalized_data_filtered['L3_rate'])

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
        cm = confusion_matrix(y_test, (model.predict(X_test) > 0.5).astype("int32"))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax1)
        st.pyplot(conf_mat_fig, use_container_width=True)

    with col2:
        try:
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            baseline_accuracy = accuracy_score(y_test, y_pred)


            # Function to compute permutation importance
            def permutation_importance(model, X_test, y_test, baseline_score, n_repeats=10):
                importances = np.zeros(X_test.shape[2])

                for i in range(X_test.shape[2]):
                    scores = np.zeros(n_repeats)
                    for n in range(n_repeats):
                        X_permuted = X_test.copy()
                        np.random.shuffle(X_permuted[:, :, i])
                        y_permuted_pred = (model.predict(X_permuted) > 0.5).astype("int32")
                        scores[n] = accuracy_score(y_test, y_permuted_pred)
                    importances[i] = baseline_score - np.mean(scores)

                return importances


            # Calculate permutation importances
            importances = permutation_importance(model, X_test, y_test, baseline_accuracy)

            fig, ax = plt.subplots()
            ax.bar(feature_names, importances)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importances for LSTM Model')
            st.pyplot(fig)
        except EOFError:
            st.write("Feature importances are not available for this model.")

with tab4:
    sliders = []
    col1, col2 = st.columns(2)
    with col1:
        for feature in feature_names:
            ing_slider = st.slider(label=feature, min_value=float(normalized_data[feature].min()),
                                   max_value=float(normalized_data[feature].max()))
            sliders.append(ing_slider)

    with col2:
        col1, col2 = st.columns(2, gap="medium")

        # Add the new data point to the buffer
        buffer.append(sliders)

        if len(buffer) >= sequence_length:
            # When buffer is full, use the last `sequence_length` points for prediction
            data = np.array(buffer[-sequence_length:])
            data = data.reshape(
                (1, sequence_length, len(feature_names)))  # Reshape the data to match the model's expected input shape
            prediction = model.predict(data)
            prediction_class = (prediction > 0.5).astype("int32")

            with col1:
                predict = "Normal"
                if target_names[prediction_class[0][0]] == '1':
                    predict = "Anomaly"
                st.markdown("Prediction:", prediction_class[0][0])
                st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(
                    predict), unsafe_allow_html=True)

            probability = prediction[0][0]

            with col2:
                st.metric(label="Model Confidence", value="{:.2f} %".format(probability * 100),
                          delta="{:.2f} %".format((probability - 0.5) * 100))

            # Define a custom prediction function for LIME
            def custom_predict(data):
                num_samples = data.shape[0]
                reshaped_data = data.reshape((num_samples, sequence_length, len(feature_names)))
                # Ensure the model returns probabilities
                probabilities = model.predict(reshaped_data)
                return np.hstack([1 - probabilities, probabilities])
