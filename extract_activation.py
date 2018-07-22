import keras
import numpy as np
import h5py
from keras.models import Model
from sklearn.metrics.pairwise import paired_distances


def main():
    x, y = load_mnist()

    score_output, prediction = get_score_and_prob(x)

    predicted_y = np.argmax(prediction, axis=1)
    prediction_correct = predicted_y == y

    labels = np.unique(y)

    with h5py.File('data/mnist_av.h5', 'w') as hf:
        for label in labels:
            label_indices = y == label
            print('Label {} has {} samples, '.format(label, np.sum(label_indices)), end='')

            label_prediction_correct = label_indices & prediction_correct
            print('{} correctly predicted'.format(np.sum(label_prediction_correct)))

            class_av = score_output[label_prediction_correct, :]
            class_mav = np.mean(class_av, axis=0, keepdims=True)
            print('   MAV shape = {}'.format(class_mav.shape))
            hf.create_dataset(f'av_{label}', data=class_av)
            hf.create_dataset(f'mav_{label}', data=class_mav)

            av_distance = np.zeros((1, class_av.shape[0]))
            for i in range(class_av.shape[0]):
                av_distance[0, i] = compute_distance(class_av[i, :].reshape(1, -1), class_mav)

            hf.create_dataset(f'av_dist_{label}', data=av_distance)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    x = x / 255.0
    x = x.reshape(-1, 28, 28, 1)
    return x, y


def get_score_and_prob(x):
    model = keras.models.load_model('models/cnn_mnist.h5')

    score_layer_model = Model(inputs=model.input, outputs=model.get_layer('score').output)
    score_output = score_layer_model.predict(x)

    prediction = model.predict(x, batch_size=4096, verbose=1)

    return score_output, prediction


def compute_distance(a, b, metric_type='cosine'):
    return paired_distances(a, b, metric=metric_type, n_jobs=1)


if __name__ == '__main__':
    main()
