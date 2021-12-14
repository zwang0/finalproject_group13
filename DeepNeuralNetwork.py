# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Deep Neural Network
# Zehua Wang, biostat625 final project

# ## Imports

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ## Data Cleaning

# +
# Date : year-month-day
# Rented Bike count - Count of bikes rented at each hour
# Hour - Hour of he day
# Temperature-Temperature in Celsius
# Humidity - %
# Windspeed - m/s
# Visibility - 10m
# Dew point temperature - Celsius
# Solar radiation - MJ/m2
# Rainfall - mm
# Snowfall - cm
# Seasons - Winter, Spring, Summer, Autumn
# Holiday - Holiday/No holiday
# Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)
# -

# load data
bike_data = pd.read_csv('data/SeoulBikeData.csv', encoding = 'unicode_escape')
# add year, month, day, and days of week
bike_data['Date'] = pd.to_datetime(bike_data['Date'])
bike_data['Year'] = pd.DatetimeIndex(bike_data['Date']).year
bike_data['Month'] = pd.DatetimeIndex(bike_data['Date']).month
bike_data['Day'] = pd.DatetimeIndex(bike_data['Date']).day
bike_data['DWeek'] = pd.DatetimeIndex(bike_data['Date']).weekday
bike_data.columns = (['Date','Rented_Bike_Count', 'Hour', 'Temperature',
                      'Humidity', 'Wind_speed', 'Visibility', 'Dew_point_temp',
                      'Solar_Rad', 'Rainfall', 'Snowfall', 'Seasons', 
                      'Holiday', 'Funct_Day','Year', 'Month', 'Day', 'DWeek'])
bike_data.head()

bike_data['Seasons'] = pd.factorize(bike_data['Seasons'])[0]
bike_data['Funct_Day'] = pd.factorize(bike_data['Funct_Day'])[0]
bike_data['Holiday'] = pd.factorize(bike_data['Holiday'])[0]

bike_data.head()

## split data into a train and test set
bike_data = bike_data.sample(frac = 1, random_state = 625).\
                reset_index(drop = True)
data_size = bike_data.shape[0]
trainset = bike_data.iloc[:round(data_size*0.7), 1:].reset_index(drop = True)
valset = bike_data.iloc[round(data_size*0.7):round(data_size*0.8), 1:].reset_index(drop = True)
testset = bike_data.iloc[round(data_size*0.8):, 1:].reset_index(drop = True)

# ## Neural Network

# data prepare
X_train = trainset.iloc[:, 1:]
y_train = trainset.iloc[:, 0]
X_val = valset.iloc[:, 1:]
y_val = valset.iloc[:, 0]
train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train = train.shuffle(len(X_train)).batch(4)
val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val = train.shuffle(len(X_train)).batch(1)

# model 1
mod1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(12, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='linear')
])
mod1.compile(
    optimizer = 'adam',
    loss = 'mean_squared_error',
)
mod1.history = mod1.fit(train, epochs = 16, validation_data = val)
mod1_loss = mod1.evaluate(val)
print('Model 1 Loss {}'.format(mod1_loss))

# model 2
mod2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])
mod2.compile(
    optimizer = 'adam',
    loss = 'mean_squared_error',
)
mod2.history = mod2.fit(train, epochs = 16, validation_data = val)
mod2_loss = mod1.evaluate(val)
print('Model 2 Loss {}'.format(mod2_loss))

print(mod2.history.history.keys())

plt.plot(mod2.history.history['loss'])
plt.plot(mod2.history.history['val_loss'])
plt.title('model MSE loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


