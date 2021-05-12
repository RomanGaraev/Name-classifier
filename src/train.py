"""
Hyper-tuning and training the models from my_models.py
NameClassifier class utilizes MyModel instances to create fully
worked binary classifier
"""
from my_models import *
from data_loader import *
from sklearn.metrics import confusion_matrix
from os.path import join
from os import pardir
import time


class NameClassifier:
    def __init__(self, model: MyModel, lr=0.001):
        """
        Wrapper over keras model. Get data from data_loader.py and the structure of model
        from ModelStructure class; provide training and testing functions for classifier
        """
        train_frame, test_frame = get_data()
        self.train_x, self.train_y = normalize(train_frame)
        self.test_x, self.test_y = normalize(test_frame)
        self.model = model
        self.classifier = self.model.get_model()
        self.classifier.compile(optimizer=keras.optimizers.Adam(lr),
                                loss=keras.losses.binary_crossentropy)

    def load(self, path):
        self.classifier = keras.models.load_model(filepath=join(pardir, "models", path))

    def train(self, epochs=100, val=0.15, save=False, save_name="model.h5", verbose=1):
        # I haven't used Tensorboard, but you can uncomment the following lines
        #log_dir = join("logs", save_name)
        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # ... callback=[tensorboard_callback]
        history = self.classifier.fit(x=self.train_x, y=self.train_y, epochs=epochs, validation_split=val,
                                      batch_size=256, verbose=verbose)
        if save:
            self.classifier.save(save_name)
        return history

    def test(self):
        # Prediction is single float value. Threshold 0.5 is needed for classification
        conf = confusion_matrix(self.test_y, np.array((self.classifier.predict(self.test_x) >= 0.5), dtype=float))
        print(f"Confusion matric {conf}, accuracy {np.trace(conf)/ sum(sum(conf))}")

    def hyper_tuning(self, model_name="model.h5"):
        # Remember the best model
        min_val_loss = 1.0
        best_weights = self.classifier
        best_params = None
        # Get the dict with model and training parameters
        for params in self.model.hyper_tuning():
            print(f"Learning rate {params['lr']}, epochs {params['epoch']}")

            self.classifier = params['model']
            optim = keras.optimizers.Adam(params['lr'])
            self.classifier.compile(optimizer=optim, loss=keras.losses.binary_crossentropy)
            history = self.train(epochs=params['epoch'], verbose=0)

            # Update the best model, if the loss was low
            last_loss = history.history['val_loss'][-1]
            print(f"Val loss:{last_loss}")
            if last_loss < min_val_loss:
                min_val_loss = last_loss
                best_weights = self.classifier
                best_params = params
        print(f"Best parameters: learning rate {best_params['lr']}, epochs {best_params['epoch']}")
        best_weights.save(model_name)
        self.classifier = best_weights
        print(self.classifier.summary())
        print("Testing...")
        print(self.test())


if __name__ == "__main__":
    print("Baseline LSTM training...")
    model = NameClassifier(BaselineLSTM())
    print(model.classifier.summary())
    t = time.time()
    model.train(save=True, save_name=join(pardir, "models", "BaselineLSTM.h5"), epochs=120)
    print("Training time:", time.time() - t)

    print("Fully-connected network training...")
    model = NameClassifier(Perceptron())
    print(model.classifier.summary())
    t = time.time()
    model.train(save=True, save_name=join(pardir, "models", "Perceptron.h5"), epochs=100)
    print("Training time:", time.time() - t)

    print("Custom LSTM training...")
    model = NameClassifier(MyLSTM())
    print(model.classifier.summary())
    t = time.time()
    model.train(save=True, save_name=join(pardir, "models", "MyLSTM.h5"), epochs=100)
    print("Training time:", time.time() - t)
