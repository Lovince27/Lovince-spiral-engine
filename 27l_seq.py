import math

# Required Sequences
def fibonacci(n):
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def lucas(n):
    seq = [2, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def primes(n):
    seq = []
    num = 2
    while len(seq) < n:
        for p in seq:
            if num % p == 0:
                break
        else:
            seq.append(num)
        num += 1
    return seq

def squares(n):
    return [i*i for i in range(n)]

def factorials(n):
    return [math.factorial(i) for i in range(n)]

# Randomness function (chaotic)
def randomness(n):
    return (n**3 + 17*n**2 + 31*n) % (2*n + 5)

# --- Generate the Lovince Sequence ---
n_terms = 30
fib = fibonacci(n_terms)
luc = lucas(n_terms)
pri = primes(n_terms)
sqr = squares(n_terms)
fac = factorials(n_terms)

lovince_sequence = []
for i in range(n_terms):
    val = (fib[i] * sqr[i]) + (pri[i] ** (luc[i] % 5 + 1)) - (fac[i] % (i + 2)) + randomness(i)
    lovince_sequence.append(val)

# --- Output the sequence ---
print("LOVINCE's AI-Generated Smart Dangerous Sequence:")
print(lovince_sequence)

# --- Graph the sequence ---
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(range(n_terms), lovince_sequence, marker='o', color='green', linestyle='-')
plt.title('LOVINCE: AI Generated Powerful Sequence')
plt.xlabel('n (index)')
plt.ylabel('Value')
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Lovince Sequence Generation (same as before)
def fibonacci(n):
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def lucas(n):
    seq = [2, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def primes(n):
    seq = []
    num = 2
    while len(seq) < n:
        for p in seq:
            if num % p == 0:
                break
        else:
            seq.append(num)
        num += 1
    return seq

def squares(n):
    return [i*i for i in range(n)]

def factorials(n):
    import math
    return [math.factorial(i) for i in range(n)]

def randomness(n):
    return (n**3 + 17*n**2 + 31*n) % (2*n + 5)

# Lovince Sequence
n_terms = 30
fib = fibonacci(n_terms)
luc = lucas(n_terms)
pri = primes(n_terms)
sqr = squares(n_terms)
fac = factorials(n_terms)

lovince_sequence = []
for i in range(n_terms):
    val = (fib[i] * sqr[i]) + (pri[i] ** (luc[i] % 5 + 1)) - (fac[i] % (i + 2)) + randomness(i)
    lovince_sequence.append(val)

# Preprocess the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(lovince_sequence).reshape(-1, 1))

# Prepare data for LSTM
X, y = [], []
time_step = 5
for i in range(len(scaled_data) - time_step):
    X.append(scaled_data[i:i+time_step, 0])
    y.append(scaled_data[i + time_step, 0])

X = np.array(X)
y = np.array(y)

# Reshaping for LSTM [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, batch_size=16)

# Predict the next 10 values (future predictions)
predictions = []
input_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
for _ in range(10):
    predicted_value = model.predict(input_sequence)
    predictions.append(predicted_value[0][0])
    input_sequence = np.append(input_sequence[:, 1:, :], predicted_value.reshape(1, 1, 1), axis=1)

# Inverse transform to get real values
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Plotting original and predicted data
plt.figure(figsize=(12, 6))
plt.plot(range(len(lovince_sequence)), lovince_sequence, label="Original Sequence", color='blue')
plt.plot(range(len(lovince_sequence), len(lovince_sequence) + 10), predictions, label="Predicted Future Sequence", color='red', linestyle='--')
plt.title('AI Generated Future Sequence (Using LSTM)')
plt.xlabel('Index (n)')
plt.ylabel('Sequence Value')
plt.legend()
plt.grid(True)
plt.show()

# Output the predicted future sequence
print("Predicted Future Sequence:")
print(predictions)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Lovince Sequence Generation (same as before)
def fibonacci(n):
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def lucas(n):
    seq = [2, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def primes(n):
    seq = []
    num = 2
    while len(seq) < n:
        for p in seq:
            if num % p == 0:
                break
        else:
            seq.append(num)
        num += 1
    return seq

def squares(n):
    return [i*i for i in range(n)]

def factorials(n):
    import math
    return [math.factorial(i) for i in range(n)]

def randomness(n):
    return (n**3 + 17*n**2 + 31*n) % (2*n + 5)

# Lovince Sequence
n_terms = 30
fib = fibonacci(n_terms)
luc = lucas(n_terms)
pri = primes(n_terms)
sqr = squares(n_terms)
fac = factorials(n_terms)

lovince_sequence = []
for i in range(n_terms):
    val = (fib[i] * sqr[i]) + (pri[i] ** (luc[i] % 5 + 1)) - (fac[i] % (i + 2)) + randomness(i)
    lovince_sequence.append(val)

# QCM Simulation (Simplified)
def qcm_surface(entropy, phi):
    """
    Simplified QCM simulation based on Entropy and Phi.
    (Replace with your actual QCM calculation)
    """
    return entropy * phi + (entropy**2) - (phi**3)  # Example function

# Create Entropy and Phi values for QCM simulation
entropy_values = np.linspace(0.1, 10, 30)  # Entropy range
phi_values = np.linspace(0.1, 1, 30)  # Phi range
entropy, phi = np.meshgrid(entropy_values, phi_values)

# Calculate QCM values
qcm = qcm_surface(entropy, phi)

# Preprocess the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(lovince_sequence).reshape(-1, 1))

# Prepare data for LSTM
X, y = [], []
time_step = 5
for i in range(len(scaled_data) - time_step):
    X.append(scaled_data[i:i+time_step, 0])
    y.append(scaled_data[i + time_step, 0])

X = np.array(X)
y = np.array(y)

# Reshaping for LSTM [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, batch_size=16)

# Predict the next 10 values (future predictions)
predictions = []
input_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
for _ in range(10):
    predicted_value = model.predict(input_sequence)
    predictions.append(predicted_value[0][0])
    input_sequence = np.append(input_sequence[:, 1:, :], predicted_value.reshape(1, 1, 1), axis=1)

# Inverse transform to get real values
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Create the 3D plot
fig = plt.figure(figsize=(14, 7))

# 3D Surface Plot (QCM)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(entropy, phi, qcm, cmap='viridis', edgecolor='none')
ax1.set_xlabel('Entropy')
ax1.set_ylabel('Phi')
ax1.set_zlabel('QCM')
ax1.set_title('Simulated QCM Surface')
fig.colorbar(surf, shrink=0.5, aspect=5)

# 2D Line Plot (Lovince Sequence + LSTM Predictions)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(range(len(lovince_sequence)), lovince_sequence, label="Original Sequence", color='blue')
ax2.plot(range(len(lovince_sequence), len(lovince_sequence) + 10), predictions, label="Predicted Future Sequence", color='red', linestyle='--')
ax2.set_title('AI Generated Future Sequence (Using LSTM)')
ax2.set_xlabel('Index (n)')
ax2.set_ylabel('Sequence Value')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Output the predicted future sequence
print("Predicted Future Sequence:")
print(predictions)
