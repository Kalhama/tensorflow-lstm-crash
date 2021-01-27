import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.python.keras import callbacks

# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

def get_train_data():
    while True:
        data_x = []
        data_y = []
        for i in range(100):
            data_x.append(np.random.rand(200,3))
            data_y.append(np.random.rand(5,1))

        yield np.array(data_x), np.array(data_y)


def build_model():
    model = Sequential()

    ### Model with LSTM crashes ###
    model.add(LSTM(200, input_shape=(200,3), return_sequences=True))
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dense(5, activation='linear'))

    print(model.summary())
    
    model.compile(loss='mse', optimizer='adam')

    print('[Model] Model Compiled')

    return model

def main():
    model = build_model()

    model.fit(
        get_train_data(),
        epochs=1000,
        steps_per_epoch=1000,
        batch_size=100
    )

if __name__ == '__main__':
    main()