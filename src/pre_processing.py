import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

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

        df_2000KHz = pd.read_parquet(os.path.join(timestamp_folder_path, "Sampling2000KHz_AEKi-0.parquet"))
        mean_2000KHz = df_2000KHz.mean().to_frame().T
        mean_2000KHz['timestamp'] = timestamp
        mean_2000KHz['label'] = label

        df_100KHz = pd.read_parquet(os.path.join(timestamp_folder_path, "Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet"))
        mean_100KHz = df_100KHz.mean().to_frame().T
        mean_100KHz['timestamp'] = timestamp
        mean_100KHz['label'] = label

        combined_100KHzdata.append(mean_100KHz)
        combined_2000KHzdata.append(mean_2000KHz)

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

def normalize_data(all_100KHzdata, all_2000KHzdata):
    features_100KHz = all_100KHzdata.drop(columns=['timestamp', 'label'])
    timestamps_100KHz = all_100KHzdata['timestamp']
    labels_100KHz = all_100KHzdata['label']

    scaler_100KHz = StandardScaler()
    normalized_features_100KHz = scaler_100KHz.fit_transform(features_100KHz)

    normalized_100KHzdata = pd.DataFrame(normalized_features_100KHz, columns=features_100KHz.columns)
    normalized_100KHzdata.insert(0, 'timestamp', timestamps_100KHz)
    normalized_100KHzdata['label'] = labels_100KHz.values

    features_2000KHz = all_2000KHzdata.drop(columns=['timestamp', 'label'])
    timestamps_2000KHz = all_2000KHzdata['timestamp']
    labels_2000KHz = all_2000KHzdata['label']

    scaler_2000KHz = StandardScaler()
    normalized_features_2000KHz = scaler_2000KHz.fit_transform(features_2000KHz)

    normalized_2000KHzdata = pd.DataFrame(normalized_features_2000KHz, columns=features_2000KHz.columns)
    normalized_2000KHzdata.insert(0, 'timestamp', timestamps_2000KHz)
    normalized_2000KHzdata['label'] = labels_2000KHz.values

    return normalized_100KHzdata, normalized_2000KHzdata

def create_labeled_sequences(data, sequence_length=10):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i+sequence_length, 1:-1].values
        label = data.iloc[i+sequence_length - 1, -1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)
