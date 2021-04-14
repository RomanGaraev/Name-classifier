"""
Testing script
"""
from train import *


def hyper_grid():
    lstm = NameClassifier(BaselineLSTM())
    print("Baseline LSTM tuning...")
    lstm.hyper_tuning(join(pardir, "models", "BaselineLSTM.h5"))

    pers = NameClassifier(Perceptron())
    print("Fully-connected network tuning...")
    pers.hyper_tuning(join(pardir, "models", "Perceptron.h5"))

    custom_lstm = NameClassifier(MyLSTM())
    print("Custom LSTM tuning...")
    custom_lstm.hyper_tuning(join(pardir, "models", "MyLSTM.h5"))


if __name__ == "__main__":
    print("Baseline LSTM testing...")
    lstm = NameClassifier(BaselineLSTM())
    lstm.load("BaselineLSTM.h5")
    lstm.test()

    print("Perceptron testing...")
    lstm = NameClassifier(Perceptron())
    lstm.load("Perceptron.h5")
    lstm.test()

    print("Custom LSTM testing...")
    lstm = NameClassifier(MyLSTM())
    lstm.load("MyLSTM.h5")
    lstm.test()
