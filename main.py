import site
import numpy as np
import pandas as pd

# function to import and condition datasets
def get_file(file_name):
    df = pd.read_csv(file_name)
    ocean_proximity_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity').astype(int)
    df = pd.concat([df.drop(['median_house_value', 'ocean_proximity' ], axis=1), ocean_proximity_dummies, df['median_house_value']], axis=1)
    df = df.dropna()
    return df

# function to calculate mean absolute error
def mae(array1,array2):
    error = np.abs(array1-array2)
    mean_error = np.mean(error)
    return mean_error

# import training dataset
train_data = np.array(get_file('training_data.csv'))

# separate parameters and prices
train_prices = train_data[:,-1]
train_params = train_data[:,:-1]

# perform linear regression
coeff = np.linalg.lstsq(train_params, train_prices, rcond=None)[0]

# import test dataset
test_data = np.array(get_file('test_data.csv'))

# seperate parameters and prices
test_prices = test_data[:,-1]
test_params = test_data[:,:-1]

# predict prices of test data
predicted_prices = np.dot(test_params,coeff)

#calculate and print the mean absolute error
avg_abs_error = mae(predicted_prices,test_prices)
print(avg_abs_error)
std_price = np.std(test_prices)
print(std_price)