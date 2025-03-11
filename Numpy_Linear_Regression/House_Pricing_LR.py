import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Numpy_Linear_Regression\housedata\data.csv', sep=',')

print(df.head())

test_data = {'x': [1, 2, 3, 4, 5], 'y': [1, 2, 3, 4, 5], 'y_hat': [1, 1.5, 2, 2.5, 3]}
test_df = pd.DataFrame(test_data)


def mean_squared_error(data, y, y_hat):
    mse = 0
    for i in range(len(data)):
        diff = (y.iloc[i] - y_hat.iloc[i]) ** 2
        mse += diff
        print(f'{i} point: {mse}')
    return mse

mse = mean_squared_error(test_df, test_df.y, test_df.y_hat)  

def calculate_linear_regression(x, a, b):
    y_hat = a*x + b
    return y_hat

def gradient_descent(data, a, b):
    predictions = {}
    for i in range(len(data)):
        y_hat = calculate_linear_regression(data.x, a, b)
        predictions[i] = y_hat
        print(predictions[i])
    print(predictions)

print(test_data.values())