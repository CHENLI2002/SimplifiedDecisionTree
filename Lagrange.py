import math

import numpy as np
from scipy.interpolate import lagrange
from sklearn.metrics import mean_squared_error

x = np.random.uniform(0, 2, 100)
y = np.sin(x)
sdv = 2

l_poly = lagrange(x, y)

x_test = np.random.uniform(0, 2, 100)
y_test = np.sin(x_test)
noise = np.random.normal(0, sdv, 100)
x_noise = x + noise
y_noise = np.sin(x_noise)

noisy_l_ploy = lagrange(x_noise, y_noise)

# since we are doing regression, I uses mse by default

training_error = math.log(mean_squared_error(y, l_poly(x)))
testing_error = math.log(mean_squared_error(y_test, l_poly(x_test)))

training_error_noisy = math.log(mean_squared_error(x_noise, noisy_l_ploy(y_noise)))
testing_error_noisy = math.log(mean_squared_error(y_test, noisy_l_ploy(x_test)))

print(f"Testing error is {testing_error}, and training error is {training_error}")

print(f"Testing error with noise is {testing_error_noisy}, and training error with noise is {training_error_noisy}")
print(f"current stv is {sdv}")
