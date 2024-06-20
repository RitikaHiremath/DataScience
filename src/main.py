import pandas as pd
import os
from sklearn.utils import shuffle
from pre_processing import preprocess_data, combine_and_interpolate_data, normalize_data, create_labeled_sequences
from model import create_lstm_model, train_model, evaluate_model
from display import plot_accuracy, plot_loss
from sklearn.model_selection import train_test_split

# Define paths to data
current_directory = os.getcwd()
ok_data_path = os.path.abspath(os.path.join(current_directory, '..', 'Data', 'OK_Measurements'))
nok_data_path = os.path.abspath(os.path.join(current_directory, '..', 'Data', 'NOK_Measurements'))

# Preprocess data
all_100KHzdata, all_2000KHzdata = preprocess_data(ok_data_path, nok_data_path)

# Combine and interpolate data
combined_data = combine_and_interpolate_data(all_100KHzdata, all_2000KHzdata)

# Normalize data
normalized_data = normalize_data(combined_data)

# Shuffle the combined data
normalized_data = shuffle(normalized_data, random_state=42)

# Create labeled sequences
sequence_length = 10
X_combined, y_combined = create_labeled_sequences(normalized_data, sequence_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

# Create LSTM model
input_shape_combined = (sequence_length, X_combined.shape[2])
model_combined = create_lstm_model(input_shape_combined)

# Train the model
history = train_model(model_combined, X_train, y_train)

# Save the model
model_combined.save('AEGuard.keras')

# Plot results
plot_accuracy(history)
plot_loss(history)

# Evaluate the model on training data
train_loss, train_accuracy, train_precision, train_recall = model_combined.evaluate(X_train, y_train)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")

# Evaluate the model on test data
test_loss, test_accuracy, test_precision, test_recall = model_combined.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Evaluate the model and print classification report
evaluate_model(model_combined, X_test, y_test)
