import keras
from keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPool2D
from keras.models import Sequential, Model

MODEL_SAVING_PATH = 'models/cnn_mnist.h5'


def start_training():
    x_train, y_train, x_test, y_test = load_mnist_data()

    model = Sequential([
        Conv2D(32, 3, padding='same', input_shape=(28, 28, 1)),
        MaxPool2D(),
        Conv2D(64, 3, padding='same'),
        MaxPool2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, name='score'),
        Activation('softmax', name='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cp_callback = keras.callbacks.ModelCheckpoint(MODEL_SAVING_PATH, save_weights_only=False, save_best_only=True, verbose=1)

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[cp_callback])


def load_mnist_data():
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test


def get_score_and_prob(x):
    model = keras.models.load_model(MODEL_SAVING_PATH)

    score_layer_model = Model(inputs=model.input, outputs=model.get_layer('score').output)
    score_output = score_layer_model.predict(x)

    prediction = model.predict(x, batch_size=4096, verbose=1)

    return score_output, prediction


if __name__ == '__main__':
    start_training()
