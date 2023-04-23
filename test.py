import numpy as np
from train import coeff

data = np.genfromtxt('train_conditioned.csv', delimiter=',', skip_header=1)

prices = data[:,-1]
params = data[:,:-1]

predicted_prices = np.dot(params,coeff)

print(prices)
print(predicted_prices)
