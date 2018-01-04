import time

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Masking, GRU, Bidirectional, RepeatVector, TimeDistributed
from keras.models import Sequential


def get_model(model_name):
    if model_name in globals():
        return globals()[model_name]
    else:
        raise ValueError("Model '{}' doesn't exist!".format(model_name))


def fit_model(model_option, x_train, y_train, w2v_model):

    start = time.time()
    model = model_option(w2v_model)
    end = time.time()
    print('Model created', (end - start))
    print(model.summary())

    early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, verbose=1, mode='auto')

    start = time.time()
    model.fit(x_train, y_train, batch_size=32, epochs=150, verbose=2, callbacks=[early_stop_callback])
    end = time.time()
    print('Fit done! {}'.format((end - start)))

    return model


# Here we defined all our models (fit, predict, summary) -> wrappers methods
# fit -> input: list of candidates with scores
# predict -> input:
class ModelBase(object):
    def __init__(self, wv, maxlen=50):
        self.wv = wv
        self.maxlen = maxlen
        self.model = None

    def summary(self):
        self.model.summary()

    def compile_model(self):
        raise Exception('Not implemented!')

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        raise Exception('Not implemented!')

    def predict(self, x):
        raise Exception('Not implemented!')


class EncGRUDecGRUSoftmax(ModelBase):
    def __init__(self, wv, maxlen=100):
        super().__init__(wv, maxlen)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        self.model = Sequential(name='EncGRUDecGRUSoftmax')
        self.model.add(Masking(mask_value=-1, input_shape=(maxlen,)))

        # Creating encoder network
        self.model.add(embedding_layer)
        self.model.add(GRU(maxlen))
        self.model.add(RepeatVector(maxlen))

        # Creating decoder network
        for _ in range(4):
            self.model.add(GRU(maxlen, return_sequences=True))
        self.model.add(TimeDistributed(Dense(4, activation='softmax')))

        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)


class EncGRUDecGRUSigmoid(ModelBase):
    def __init__(self, wv, maxlen=100):
        super().__init__(wv, maxlen)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        self.model = Sequential(name='EncGRUDecGRUSigmoid')
        self.model.add(Masking(mask_value=-1, input_shape=(maxlen,)))

        # Creating encoder network
        self.model.add(embedding_layer)
        self.model.add(GRU(maxlen))
        self.model.add(RepeatVector(maxlen))

        # Creating decoder network
        for _ in range(4):
            self.model.add(GRU(maxlen, return_sequences=True))
        self.model.add(TimeDistributed(Dense(4, activation='sigmoid')))

        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)


class EncBiGRUDecGRUSoftmax(ModelBase):
    def __init__(self, wv, maxlen=100):
        super().__init__(wv, maxlen)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        self.model = Sequential(name='EncBiGRUDecGRUSoftmax')
        self.model.add(Masking(mask_value=-1, input_shape=(maxlen,)))

        # Creating encoder network
        self.model.add(embedding_layer)
        self.model.add(Bidirectional(GRU(maxlen)))
        self.model.add(RepeatVector(maxlen))

        # Creating decoder network
        for _ in range(4):
            self.model.add(GRU(maxlen, return_sequences=True))
        self.model.add(TimeDistributed(Dense(4, activation='softmax')))

        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)


class EncBiDecGRUSigmoid(ModelBase):
    def __init__(self, wv, maxlen=100):
        super().__init__(wv, maxlen)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        self.model = Sequential(name='EncBiDecGRUSigmoid')
        self.model.add(Masking(mask_value=-1, input_shape=(maxlen,)))

        # Creating encoder network
        self.model.add(embedding_layer)
        self.model.add(Bidirectional(GRU(maxlen)))
        self.model.add(RepeatVector(maxlen))

        # Creating decoder network
        for _ in range(4):
            self.model.add(GRU(maxlen, return_sequences=True))
        self.model.add(TimeDistributed(Dense(4, activation='sigmoid')))

        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)


class EncGRUDecBiGRUSoftmax(ModelBase):
    def __init__(self, wv, maxlen=100):
        super().__init__(wv, maxlen)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        self.model = Sequential(name='EncGRUDecBiGRUSoftmax')
        self.model.add(Masking(mask_value=-1, input_shape=(maxlen,)))

        # Creating encoder network
        self.model.add(embedding_layer)
        self.model.add(GRU(maxlen))
        self.model.add(RepeatVector(maxlen))

        # Creating decoder network
        for _ in range(4):
            self.model.add(Bidirectional(GRU(maxlen, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(4, activation='softmax')))

        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)


class EncGRUDecBiGRUSigmoid(ModelBase):
    def __init__(self, wv, maxlen=100):
        super().__init__(wv, maxlen)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        self.model = Sequential(name='EncGRUDecBiGRUSigmoid')
        self.model.add(Masking(mask_value=-1, input_shape=(maxlen,)))

        # Creating encoder network
        self.model.add(embedding_layer)
        self.model.add(Bidirectional(GRU(maxlen)))
        self.model.add(RepeatVector(maxlen))

        # Creating decoder network
        for _ in range(4):
            self.model.add(Bidirectional(GRU(maxlen, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(4, activation='sigmoid')))

        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)


class EncBiGRUDecBiGRUSoftmax(ModelBase):
    def __init__(self, wv, maxlen=100):
        super().__init__(wv, maxlen)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        self.model = Sequential(name='EncBiGRUDecBiGRUSoftmax')
        self.model.add(Masking(mask_value=-1, input_shape=(maxlen,)))

        # Creating encoder network
        self.model.add(embedding_layer)
        self.model.add(Bidirectional(GRU(maxlen)))
        self.model.add(RepeatVector(maxlen))

        # Creating decoder network
        for _ in range(4):
            self.model.add(Bidirectional(GRU(maxlen, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(4, activation='softmax')))

        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)


class EncBiGRUDecBiGRUSigmoid(ModelBase):
    def __init__(self, wv, maxlen=100):
        super().__init__(wv, maxlen)
        embedding_layer = self.wv.model.wv.get_embedding_layer()

        self.model = Sequential(name='EncBiGRUDecBiGRUSigmoid')
        self.model.add(Masking(mask_value=-1, input_shape=(maxlen,)))

        # Creating encoder network
        self.model.add(embedding_layer)
        self.model.add(Bidirectional(GRU(maxlen)))
        self.model.add(RepeatVector(maxlen))

        # Creating decoder network
        for _ in range(4):
            self.model.add(Bidirectional(GRU(maxlen, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(4, activation='sigmoid')))

        self.compile_model()

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    def fit(self, x, y, batch_size, epochs, verbose, callbacks):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)