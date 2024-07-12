import pandas as pd
import os
from pre_processing import preprocess_data, combine_data, normalize_data, create_labeled_sequences
from model import create_lstm_model, train_model, evaluate_model
from display import plot_accuracy, plot_loss
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Define paths to data
current_directory = os.getcwd()
ok_data_path = os.path.abspath(os.path.join(current_directory, 'OK_Measurements'))
nok_data_path = os.path.abspath(os.path.join(current_directory , 'NOK_Measurements'))

# Preprocess data
all_100KHzdata, all_2000KHzdata = preprocess_data(ok_data_path, nok_data_path)

# Combine and interpolate data
combined_data = combine_data(all_100KHzdata, all_2000KHzdata)
combined_data = combined_data.sort_values(by='timestamp').reset_index(drop=True)

# Normalize data
normalized_data = normalize_data(combined_data)

# Create labeled sequences
sequence_length = 10
X_combined, y_combined = create_labeled_sequences(normalized_data, sequence_length)



# Main script (modified part only)
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, stratify=y_combined, random_state=42)


# Reshape X_train to 2D for SMOTE
n_samples, seq_len, n_features = X_train.shape
X_train_reshaped = X_train.reshape((n_samples, seq_len * n_features))

# Apply SMOTE to the reshaped data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)

# Reshape X_train_resampled back to 3D
X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], seq_len, n_features))

# Create LSTM model
input_shape_combined = (sequence_length, X_combined.shape[2])
model_combined = create_lstm_model(input_shape_combined)

# Train the model
history = train_model(model_combined, X_train_resampled, y_train_resampled)

# Save the model
model_combined.save('AEGuard.keras')

# Plot results
plot_accuracy(history)
plot_loss(history)

# Evaluate the model on training data
train_loss, train_accuracy, train_precision, train_recall, train_f1 = evaluate_model(model_combined, X_train, y_train)
print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy*100:.2f}%, Training Precision: {train_precision:.2f}, Training Recall: {train_recall:.2f}, Training F1 Score: {train_f1:.2f}")

# Evaluate the model on test data
test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model_combined, X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%, Test Precision: {test_precision:.2f}, Test Recall: {test_recall:.2f}, Test F1 Score: {test_f1:.2f}")

# Create a results table
results = {
    "Dataset": ["Training", "Test"],
    "Loss": [train_loss, test_loss],
    "Accuracy": [train_accuracy, test_accuracy],
    "Precision": [train_precision, test_precision],
    "Recall": [train_recall, test_recall],
    "F1 Score": [train_f1, test_f1]
}

results_df = pd.DataFrame(results)
print("\nResults Summary:")
print(results_df)
