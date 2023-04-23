import numpy as np

data = np.genfromtxt('train_conditioned.csv', delimiter=',', skip_header=1)

prices = data[:,-1]
params = data[:,:-1]

coeff = np.linalg.lstsq(params, prices, rcond=None)[0]

