"""
Collections of keras classifiers. MyModel class describes the structure
of model and hyper-tuning generator for it
"""
import tensorflow.keras as keras


class MyModel:
    """
    Abstract class for models structure

    Contains some common params for hyper-tuning
    """
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    epochs = [20, 50, 100, 120]
    embedding_size = [5, 10]

    def get_model(self):
        raise NotImplementedError

    def hyper_tuning(self):
        raise NotImplementedError


class BaselineLSTM(MyModel):
    def __init__(self):
        self.neurons = [3, 5, 10, 15]

    def get_model(self, neurons=15, emb=10):
        """
        Simple one layer LSTM
        """
        model = keras.Sequential()
        model.add(keras.layers.Embedding(200, emb, input_length=15))
        model.add(keras.layers.LSTM(neurons, return_sequences=True))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
        return model

    def hyper_tuning(self):
        for lr in self.learning_rates:
            for neuron in self.neurons:
                for epoch in self.epochs:
                    for emb in self.embedding_size:
                        yield {"lr": lr, "epoch": epoch, "model": self.get_model(neurons=neuron, emb=emb)}


class Perceptron(MyModel):
    def __init__(self):
        self.first_layer = [32, 64]
        self.second_layer = [8, 16, 32]
        self.flattens = [keras.layers.GlobalMaxPool1D(), keras.layers.GlobalAveragePooling1D(), keras.layers.Flatten()]

    def get_model(self, flatten_layer=keras.layers.Flatten(), emb=10, n1=32, n2=8):
        """
        Classifier with dense layers only, without recurrent cells
        """
        model = keras.Sequential()
        model.add(keras.layers.Embedding(200, emb, input_length=15))
        model.add(flatten_layer)
        model.add(keras.layers.Dense(n1, activation=keras.activations.relu))
        model.add(keras.layers.Dense(n2, activation=keras.activations.relu))
        model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
        return model

    def hyper_tuning(self):
        for lr in self.learning_rates:
            for n1 in self.first_layer:
                for n2 in self.second_layer:
                    for epoch in self.epochs:
                        for flatten in self.flattens:
                            for emb in self.embedding_size:
                                yield {"lr": lr, "epoch": epoch,
                                       "model": self.get_model(n1=n1, n2=n2, flatten_layer=flatten,emb=emb)}


class MyLSTM(MyModel):
    def __init__(self):
        self.first_layer = [16, 32, 64]
        self.LSTM_size = [5, 10, 15]

    def get_model(self, n1=32, n_lstm=15, emb=10):
        """
        Custom classifier, combining LSTM for feature extraction and dense layers for precise classification
        """
        model = keras.Sequential()
        model.add(keras.layers.Embedding(200, emb, input_length=15))
        model.add(keras.layers.LSTM(n_lstm, return_sequences=True))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(n1, activation=keras.activations.relu))
        model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
        return model

    def hyper_tuning(self):
        for lr in self.learning_rates:
            for n1 in self.first_layer:
                for emb in self.embedding_size:
                    for epoch in self.epochs:
                        for lstm_neurons in self.LSTM_size:
                            yield {"lr": lr, "epoch": epoch,
                                   "model": self.get_model(n1=n1, n_lstm=lstm_neurons, emb=emb)}
