import numpy as np
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(dataset, w2v_model):
    x = np.array([[w2v_model.word2idx(token) for token in row[0][::-1]] for row in dataset])
    x = sequence.pad_sequences(x, maxlen=99, value=-1)
    x = sequence.pad_sequences(x, maxlen=100, value=-1)

    label_encoder = LabelEncoder()
    label_encoder.fit(['O', 'B', 'I'])

    y = [label_encoder.transform(row[1]) for row in dataset]
    y = sequence.pad_sequences(y, maxlen=100, padding='post', truncating='post', value=-1)
    y = np.array([np.eye(4)[elem] for elem in y])

    return x, y, label_encoder


def reverse_preprocessing_y(y, label_encoder):
    y_iob = [[label_encoder.inverse_transform(np.where(token == 1)[0][0]) for token in row
              if np.where(token == 1)[0][0] != 3] for row in y]

    return iob_2_conditions(y_iob)


def iob_2_conditions(y_iob):
    y_indexes = []
    for row in y_iob:
        conds = []
        cond = []
        for index, elem in enumerate(row):
            if elem == 'B':
                if len(cond) > 0:
                    conds.append(cond)
                cond = [index + 1]
            elif elem == 'I':
                cond.append(index + 1)
            else:
                if len(cond) > 0:
                    conds.append(cond)
                cond = []
        y_indexes.append(conds)
    return y_indexes
