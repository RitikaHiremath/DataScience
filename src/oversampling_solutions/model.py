from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from sklearn.metrics import  precision_score, recall_score, f1_score
from tensorflow.keras.losses import BinaryCrossentropy

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape, kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(LSTM(32, kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
    return model


def train_model(model, X_train, y_train):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
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
    
    return loss, accuracy, precision, recall, f1
