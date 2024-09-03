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

    print("Parámetros del modelo: w = {:.1f}, b = {:.1f}".format(w[0][0], b[0]))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.xlabel("epoch")
    plt.ylabel("Error cuadrático medio")
    plt.title("ECM vs. epochs")

    x_min = x.min()
    x_max = x.max()
    n_x = normalize_x(x, x_min, x_max)
    y_regr = y.min() + model.predict(n_x) * (y.max() - y.min())
    plt.subplot(1, 2, 2)
    plt.scatter(x, y)
    plt.plot(x, y_regr, "r")
    plt.title("Datos originales y Regresión lineal")
    plt.show()

    print("Predicciones:")
    print("============")

    predict_150 = predict(model, 150, y, x_min, x_max)
    predict_160 = predict(model, 160, y, x_min, x_max)
    predict_170 = predict(model, 170, y, x_min, x_max)
    predict_180 = predict(model, 180, y, x_min, x_max)
    predict_190 = predict(model, 190, y, x_min, x_max)
    predict_200 = predict(model, 200, y, x_min, x_max)

    print("Predicción para altura de 150 cm es de {:.2f} metros.".format(predict_150))
    print("Predicción para altura de 160 cm es de {:.2f} metros.".format(predict_160))
    print("Predicción para altura de 170 cm es de {:.2f} metros.".format(predict_170))
    print("Predicción para altura de 180 cm es de {:.2f} metros.".format(predict_180))
    print("Predicción para altura de 190 cm es de {:.2f} metros.".format(predict_190))
    print("Predicción para altura de 200 cm es de {:.2f} metros.".format(predict_200))


def predict(model, value, y, min_x, max_x):
    x_pred = np.array([value]).reshape(-1, 1)
    x_pred_norm = normalize_x(x_pred, min_x, max_x)
    y_pred = y.min() + model.predict(x_pred_norm) * (y.max() - y.min())

    return y_pred[0][0]


def normalize_x(x: np.array, min_x, max_x) -> np.array:
    return (x - min_x) / (max_x - min_x)

def normalize_y(y):
    return (y - y.min()) / (y.max() - y.min())

def train_model(x, y, epochs=EPOCHS_OF_TRAINING):

    # np.random.seed(2)
    keras.utils.set_random_seed(2)
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation="linear"))

    sgd = SGD(learning_rate=0.0004)

    # calcular el mínimo y máximo de x y y y normalizar los datos entre 0 y 1
    
    x_min = x.min()
    x_max = x.max()
    n_x = normalize_x(x, x_min, x_max)
    n_y = normalize_y(y)

    model.compile(optimizer=sgd, loss="mse")

    batch_size = len(x)

    history = model.fit(n_x, n_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, history


if __name__ == "__main__":
    main()
