# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set & testing set
dataset_train = pd.read_csv('training.csv')
training_set = dataset_train.iloc[:, 1:5].values
testing_set = dataset_train.iloc[:, 3:4].values

#counting rows
rows, columns = training_set.shape

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.fit_transform(testing_set)

# Getting the inputs and the outputs
X_train = training_set_scaled[0:rows-1,:]
y_train = testing_set_scaled[1:rows, :]

#Reshaping
X_train = np.reshape(X_train, (rows-1, 1, 4))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU

# Initialising the RNN
regressor = Sequential()

#Adding the input layer and the LSTM layer
regressor.add(GRU(units = 4, activation = 'sigmoid', input_shape = (1, 4)))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compailing the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs =200)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('testing.csv')
rows_t, columns_t = dataset_test.shape
real_stock_price = dataset_test.iloc[0:rows_t-1, 1:5].values
real_stock_price_output = dataset_test.iloc[1:rows_t, 3:4].values
real_stock_price_output_df = pd.DataFrame(real_stock_price_output)
real_stock_price_output_df.to_csv('real_stock_price_output.csv')
# Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)
#real_stock_price_output = inputs[:, 1:2]
inputs = np.reshape(inputs, (rows_t-1, 1, 4))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price_df = pd.DataFrame(predicted_stock_price)
predicted_stock_price_df.to_csv('predicted_stock_price.csv')

# Visualising the results
plt.plot(real_stock_price_output, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
plt.savefig('fig.png')

#Part 4 - Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error
mean_real_stock_price_output_df = real_stock_price_output_df.mean()
rmse = math.sqrt(mean_squared_error(real_stock_price_output, predicted_stock_price))
rmse_percentage = rmse/mean_real_stock_price_output_df







