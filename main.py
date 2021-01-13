import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.python.keras import callbacks

# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

def get_train_data():
    data_x = []
    data_y = []
    for i in range(8000):
        data_x.append(np.random.rand(200,3))
        data_y.append(np.random.rand(5,1))

    return np.array(data_x), np.array(data_y)


def build_model():
    model = Sequential()

    ### Model with LSTM crashes ###
    model.add(LSTM(200, input_shape=(200,3), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='linear'))

    ### Model without LTSM doesn't crash ###
    # model.add(Flatten(input_shape=(200,3)))
    # model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(5, activation='linear'))
    
    model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # print(model.summary())
    
    print('[Model] Model Compiled')

    return model

def train_model(model, x, y):
    print('[Model] Training Started')

    model.fit(
        x,
        y,
        epochs=1000,
        batch_size=32
    )

    print('[Model] Training Completed')

def main():
    x, y = get_train_data()

    model = build_model()

    train_model(
        model,
        x,
        y
    )



if __name__ == '__main__':
    main()