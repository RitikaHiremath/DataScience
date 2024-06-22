from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from loss_function import CustomLoss
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape, kernel_regularizer='l2'))
    model.add(Dropout(0.2))
    model.add(LSTM(50, kernel_regularizer='l2'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=CustomLoss(), metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model

def train_model(model, X_combined, y_combined):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_combined, y_combined,
        epochs=50,
        batch_size=8,
        validation_split=0.3,
        callbacks=[early_stopping]
    )
    return history

def train_model(model, X_combined, y_combined):
    # Shuffle the data to ensure proper training
    X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_combined, y_combined,
        epochs=50,  # Reduced number of epochs
        batch_size=16,  # Increased batch size to reduce noise
        validation_split=0.3,
        callbacks=[early_stopping]
    )
    
    return history

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_classes = (y_pred > 0.5).astype("int32")
    
    # Evaluate the model to get loss and accuracy
    evaluation_metrics = model.evaluate(X, y, verbose=0)
    loss = evaluation_metrics[0]
    accuracy = evaluation_metrics[1]
    
    # Calculate precision, recall, and F1 score
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
    
    return loss, accuracy, precision, recall, f1

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     y_pred_classes = (y_pred > 0.5).astype("int32")
#     print(classification_report(y_test, y_pred_classes))
    
#     # Plot confusion matrix
#     cm = confusion_matrix(y_test, y_pred_classes)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
#     disp.plot(cmap=plt.cm.Blues)
#     plt.show()
