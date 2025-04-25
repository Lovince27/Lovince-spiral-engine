import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Spiral ML Model (TensorFlow DNN)
class SpiralMLModel:
    def __init__(self, amplitude=40.5, phi=(1 + np.sqrt(5)) / 2, spiral_theta_range=4 * np.pi):
        self.amplitude = amplitude
        self.phi = phi
        self.spiral_theta_range = spiral_theta_range
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def generate_training_data(self):
        theta = np.linspace(0, self.spiral_theta_range, 1000)
        b = np.log(self.phi) / (np.pi / 2)
        r_creation = self.amplitude * np.exp(-b * theta) + np.random.normal(0, 0.1, theta.size)
        r_collapse = self.amplitude * np.exp(b * theta) + np.random.normal(0, 0.1, theta.size)
        X = np.vstack((theta, np.ones_like(theta), np.sin(theta))).T
        y = np.concatenate([r_creation, r_collapse])
        return X, y

    def train(self, epochs=50):
        X, y = self.generate_training_data()
        self.model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=1)

    def predict(self, theta):
        X = np.vstack((theta, np.ones_like(theta), np.sin(theta))).T
        return self.model.predict(X).flatten()

# Usage
spiral_model = SpiralMLModel()
spiral_model.train(epochs=50)
theta_test = np.linspace(0, 4 * np.pi, 100)
predictions = spiral_model.predict(theta_test)
