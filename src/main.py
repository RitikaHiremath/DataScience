import pandas as pd
import os
from pre_processing import preprocess_data, normalize_data, create_labeled_sequences
from model import create_lstm_model, train_model, evaluate_model
from display import plot_accuracy, plot_loss

# Define paths to data
current_directory = os.getcwd()
ok_data_path = os.path.abspath(os.path.join(current_directory, '..', 'Data', 'OK_Measurements'))
nok_data_path = os.path.abspath(os.path.join(current_directory, '..', 'Data', 'NOK_Measurements'))

# Preprocess data
all_100KHzdata, all_2000KHzdata = preprocess_data(ok_data_path, nok_data_path)

# Normalize data
normalized_100KHzdata, normalized_2000KHzdata = normalize_data(all_100KHzdata, all_2000KHzdata)

# Concatenate data
normalized_100KHzdata = normalized_100KHzdata.set_index('timestamp')
normalized_2000KHzdata = normalized_2000KHzdata.set_index('timestamp')

combined_data = pd.concat([normalized_100KHzdata, normalized_2000KHzdata], axis=1, join='inner').reset_index()
combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

label = combined_data.pop('label')
combined_data['label'] = label

# Create labeled sequences
sequence_length = 10
X_combined, y_combined = create_labeled_sequences(combined_data, sequence_length)

# Create LSTM model
input_shape_combined = (sequence_length, X_combined.shape[2])
model_combined = create_lstm_model(input_shape_combined)

# Train the model
history = train_model(model_combined, X_combined, y_combined)

# Save the model
model_combined.save('AEGuard.h5')

# Evaluate the model
X_test = X_combined  # Ideally, split your data into training and testing sets
y_test = y_combined  # Ideally, split your data into training and testing sets
evaluate_model(model_combined, X_test, y_test)

# Plot results
plot_accuracy(history)
plot_loss(history)
