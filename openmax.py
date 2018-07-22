import numpy as np
from sklearn.metrics.pairwise import paired_distances

from evt import query_weibull


def compute_distance(a, b):
    return paired_distances(a, b, metric="cosine", n_jobs=1)


def recalibrate_scores(activation_vector, weibull_model, labels, alpha_rank=10):
    ranked_list = activation_vector.argsort().ravel()[::-1]
    alpha_weights = [((alpha_rank + 1) - i) / float(alpha_rank) for i in range(1, alpha_rank + 1)]
    ranked_alpha = np.zeros((len(labels)))

    for i in range(alpha_rank):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate score for each class to include probability of unknown
    openmax_score = []
    openmax_score_unknown = []

    for label_index, label in enumerate(labels):
        # get distance between current channel and mean vector
        weibull = query_weibull(label, weibull_model)
        av_distance = compute_distance(activation_vector.reshape(1, -1), weibull[0][0])

        # obtain w_score for the distance and compute probability of the distance being unknown wrt to mean training vector
        wscore = weibull[2][0].w_score(av_distance)
        modified_score = activation_vector[label_index] * (1 - wscore * ranked_alpha[label_index])
        openmax_score += [modified_score]
        openmax_score_unknown += [activation_vector[label_index] - modified_score]

    openmax_score = np.array(openmax_score)
    openmax_score_unknown = np.array(openmax_score_unknown)

    # Pass the re-calibrated scores for the image into OpenMax
    openmax_probab = compute_openmax_probability(openmax_score, openmax_score_unknown, labels)
    softmax_probab = compute_softmax_probability(activation_vector)  # Calculate SoftMax ???
    return np.array(openmax_probab), softmax_probab


def compute_softmax_probability(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def compute_openmax_probability(openmax_score, openmax_score_unknown, labels):
    exp_scores = []
    for label_index, label in enumerate(labels):
        exp_scores += [np.exp(openmax_score[label_index])]

    total_denominator = np.sum(np.exp(openmax_score)) + np.exp(np.sum(openmax_score_unknown))
    prob_scores = np.array(exp_scores) / total_denominator
    prob_unknown = np.exp(np.sum(openmax_score_unknown)) / total_denominator

    return prob_scores.tolist() + [prob_unknown]


# -------------------------------------------------------------------------------------------------

def recalibrate_scores_custom(activation_vector, softmax_score, weibull_model, labels, alpha_rank=10):
    ranked_list = softmax_score.argsort().ravel()[::-1]
    alpha_weights = [((alpha_rank + 1) - i) / float(alpha_rank) for i in range(1, alpha_rank + 1)]
    ranked_alpha = np.zeros((len(labels)))

    for i in range(alpha_rank):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate score for each class to include probability of unknown
    openmax_score = []
    openmax_score_unknown = []

    for label_index, label in enumerate(labels):
        # get distance between current channel and mean vector
        weibull = query_weibull(label, weibull_model)
        av_distance = compute_distance(activation_vector.reshape(1, -1), weibull[0][0])

        # obtain w_score for the distance and compute probability of the distance being unknown wrt to mean training vector
        wscore = weibull[2][0].w_score(av_distance)
        modified_score = softmax_score[label_index] * (1 - wscore * ranked_alpha[label_index])
        openmax_score += [modified_score]
        openmax_score_unknown += [softmax_score[label_index] - modified_score]

    openmax_score = np.array(openmax_score)
    openmax_score_unknown = np.array(openmax_score_unknown)

    # Pass the re-calibrated scores for the image into OpenMax
    openmax_probab = compute_openmax_probability(openmax_score, openmax_score_unknown, labels)
    softmax_probab = compute_softmax_probability(softmax_score)  # Calculate SoftMax ???
    return np.array(openmax_probab), softmax_probab