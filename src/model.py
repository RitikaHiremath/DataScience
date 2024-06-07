from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from loss_function import CustomLoss

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=CustomLoss(), metrics=['accuracy'])
    return model

def train_model(model, X_combined, y_combined):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_combined, y_combined,
        epochs=50,
        batch_size=8,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    return history
