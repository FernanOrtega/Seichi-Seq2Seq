import numpy as np
from collections import Counter

from dataset_preprocessor import reverse_preprocessing_y, iob_2_conditions


def measure_macro(l_conf_matrices, function_micro):
    measure_per_matrix = [function_micro(conf_matrix) for conf_matrix in l_conf_matrices]
    return sum(measure_per_matrix) / len(measure_per_matrix)


def f1_micro(conf_matrix):
    dividend = 2 * conf_matrix['tp']
    divisor = 2 * conf_matrix['tp'] + conf_matrix['fp'] + conf_matrix['fn']

    return 1.0 if dividend == 0.0 and divisor == 0.0 else dividend / divisor


def precision_micro(conf_matrix):
    dividend = conf_matrix['tp']
    divisor = conf_matrix['tp'] + conf_matrix['fp']

    return 1.0 if dividend == 0.0 and divisor == 0.0 else dividend / divisor


def recall_micro(conf_matrix):
    dividend = conf_matrix['tp']
    divisor = conf_matrix['tp'] + conf_matrix['fn']

    return 1.0 if dividend == 0.0 and divisor == 0.0 else dividend / divisor


def tnr_micro(conf_matrix):
    dividend = conf_matrix['tn']
    divisor = conf_matrix['tn'] + conf_matrix['fp']

    return 1.0 if dividend == 0.0 and divisor == 0.0 else dividend / divisor


def validate(l_predicted, l_expected, l_sizes):
    # confusion matrix
    l_conf_matrices = []
    l_conf_matrices_t = []
    for predicted, expected, size_sentence in zip(l_predicted, l_expected, l_sizes):
        flat_predicted = [index for elem in predicted for index in elem]
        flat_expected = [index for elem in expected for index in elem]
        # Chunk based computation
        tp = sum([len([index for index in elem if index in flat_expected]) / len(elem) for elem in predicted])
        fp = sum([len([index for index in elem if index not in flat_expected]) / len(elem) for elem in predicted])
        fn = sum([len([index for index in elem if index not in flat_predicted]) / len(elem) for elem in expected])
        tn = sum([len([index for index in elem if index in flat_predicted]) / len(elem) for elem in expected])
        l_conf_matrices.append({'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})

        # Token based computation
        tp = len(set(flat_predicted) & set(flat_expected))
        fp = len(flat_predicted) - tp
        fn = len(flat_expected) - tp
        tn = size_sentence - tp - fp - fn
        l_conf_matrices_t.append({'tp': tp / size_sentence, 'tn': tn / size_sentence, 'fp': fp / size_sentence,
                                  'fn': fn / size_sentence})

    confusion_matrix = Counter()
    for d in l_conf_matrices:
        for k, v in d.items():
            confusion_matrix[k] += v
    confusion_matrix_t = Counter()
    for d in l_conf_matrices_t:
        for k, v in d.items():
            confusion_matrix_t[k] += v

    return [measure_macro(l_conf_matrices, precision_micro), measure_macro(l_conf_matrices, recall_micro),
            measure_macro(l_conf_matrices, f1_micro), measure_macro(l_conf_matrices, tnr_micro),
            precision_micro(confusion_matrix), recall_micro(confusion_matrix), f1_micro(confusion_matrix),
            tnr_micro(confusion_matrix),
            measure_macro(l_conf_matrices_t, precision_micro), measure_macro(l_conf_matrices_t, recall_micro),
            measure_macro(l_conf_matrices_t, f1_micro), measure_macro(l_conf_matrices_t, tnr_micro),
            precision_micro(confusion_matrix_t), recall_micro(confusion_matrix_t), f1_micro(confusion_matrix_t),
            tnr_micro(confusion_matrix_t)]


def evaluate(x_test, y_test, l_sizes, model, label_encoder):

    y_pred = model.predict(x_test)
    y_pred = [[np.eye(4)[np.argmax(token)] for token in row] for row in y_pred]
    y_pred = reverse_preprocessing_y(y_pred, label_encoder)
    y_exp = iob_2_conditions(y_test)

    return validate(y_pred, y_exp, l_sizes)