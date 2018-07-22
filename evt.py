import libmr

import numpy as np
from sklearn.metrics.pairwise import paired_distances


def weibull_fit_tails(av_map, tail_size=200, metric_type='cosine'):
    weibull_model = {}
    labels = av_map.keys()

    for label in labels:
        print(f'EVT fitting for label {label}')
        weibull_model[label] = {}

        class_av = av_map[label]
        class_mav = np.mean(class_av, axis=0, keepdims=True)

        av_distance = np.zeros((1, class_av.shape[0]))
        for i in range(class_av.shape[0]):
            av_distance[0, i] = compute_distance(class_av[i, :].reshape(1, -1), class_mav, metric_type=metric_type)

        weibull_model[label]['mean_vec'] = class_mav
        weibull_model[label]['distances'] = av_distance

        mr = libmr.MR()

        tail_size_fix = min(tail_size, av_distance.shape[1])
        tails_to_fit = sorted(av_distance[0, :])[-tail_size_fix:]
        mr.fit_high(tails_to_fit, tail_size_fix)

        weibull_model[label]['weibull_model'] = mr

    return weibull_model


def compute_distance(a, b, metric_type):
    return paired_distances(a, b, metric=metric_type, n_jobs=1)


def query_weibull(label, weibull_model):
    return [
        [weibull_model[label]['mean_vec']],
        [weibull_model[label]['distances']],
        [weibull_model[label]['weibull_model']]
    ]
