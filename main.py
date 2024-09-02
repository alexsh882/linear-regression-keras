import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# imports keras
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD
import keras


CSV_PATH = "./altura_peso.csv"
EPOCHS_OF_TRAINING = 500


def read_data(path=CSV_PATH):
    data = pd.read_csv(path, sep=",", names=["Altura", "Peso"], skiprows=1)
    return data


def main():
    # Load the data
    data = read_data()

    x = data["Altura"]
    y = data["Peso"]

    # Create the model
    model, history = train_model(x, y, 5000)

    capas = model.layers[0]
    w, b = capas.get_weights()

    print('Parámetros del modelo: w = {:.1f}, b = {:.1f}'.format(w[0][0], b[0]))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.xlabel("epoch")
    plt.ylabel("Error cuadrático medio")
    plt.title("ECM vs. epochs")

    n_x = (x - x.min()) / (x.max() - x.min())
    y_regr = y.min() + model.predict(n_x) * (y.max() - y.min())
    plt.subplot(1, 2, 2)
    plt.scatter(x, y)
    plt.plot(x, y_regr, "r")
    plt.title("Datos originales y Regresión lineal")
    plt.show()

    



def train_model(x, y, epochs=EPOCHS_OF_TRAINING):

    
    # np.random.seed(2)
    keras.utils.set_random_seed(2)
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation="linear"))

    sgd = SGD(learning_rate=0.0004)

    # calcular el mínimo y máximo de x y y y normalizar los datos entre 0 y 1
    n_x = (x - x.min()) / (x.max() - x.min())
    n_y = (y - y.min()) / (y.max() - y.min())

    model.compile(optimizer=sgd, loss="mse")

    batch_size = len(x)

    history = model.fit(n_x, n_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, history


if __name__ == "__main__":
    main()
