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

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# path
ROOT_DIR = os.path.abspath("../")

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
data_path = os.path.join(ROOT_DIR, 'data/SeoulBikeData.csv')
bike_data = pd.read_csv(data_path, encoding = 'unicode_escape')
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
# bike_data['Rented_Bike_Count'] = bike_data['Rented_Bike_Count'].astype(float)

bike_data.head()

data_size = bike_data.shape[0]
idx =  list(range(data_size))
np.random.seed(48107)
np.random.shuffle(idx)
train_idx = idx[:int(data_size*0.8)]
test_idx = idx[int(data_size*0.8):int(data_size*0.9)]
val_idx = idx[int(data_size*0.9):]

## split data into a train and test set
trainset = bike_data.iloc[train_idx, 1:].reset_index(drop = True)
valset = bike_data.iloc[val_idx, 1:].reset_index(drop = True)
testset = bike_data.iloc[test_idx, 1:].reset_index(drop = True)

# ## Neural Network

CURRENT_DIR = os.path.join(ROOT_DIR, "NeuralNetwork")
WEIGHT_DIR = os.path.join(CURRENT_DIR, "weights")

# data prepare
X_train = trainset.iloc[:, 1:]
y_train = trainset.iloc[:, 0]
X_test = testset.iloc[:, 1:]
y_test = testset.iloc[:, 0]
X_val = valset.iloc[:, 1:]
y_val = valset.iloc[:, 0]


# +
# model 1
def create_model1():  
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error',
    )
    return model

# model 2
def create_model2():  
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error',
    )
    return model

# model 3
def create_model3():  
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error',
    )
    return model

# model 4
def create_model4():  
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error',
    )
    return model

# model 5
def create_model5():  
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu',
                             activity_regularizer=tf.keras.regularizers.L1(0.1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu',
                             activity_regularizer=tf.keras.regularizers.L1(0.1)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu',
                             activity_regularizer=tf.keras.regularizers.L1(0.1)),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error',
    )
    return model

# model 6
def create_model6():  
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer = 'adam',
        loss = 'mean_squared_error',
    )
    return model


# -

mod = create_model6()
mod.summary()
mod_path = "mod6/cp.ckpt"
mod_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(WEIGHT_DIR, mod_path),
    save_weights_only = True,
    monitor = 'val_loss',
    mode = "min",
    verbose = 0,
    save_best_only = True
)
mod_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = "min",
    patience = 100,
    restore_best_weights = True
)

mod_history = mod.fit(
    X_train, y_train,
    batch_size = 64, # default batch_size = 32
    epochs = 1000, 
    callbacks = [mod_earlystop, mod_checkpoint],
    validation_data = (X_val, y_val),
    verbose = 2
)
mod_loss = mod.evaluate(X_val, y_val)
print('Model Loss {}'.format(mod_loss))

mod.evaluate(X_test, y_test) # test mse loss

plt.plot(mod_history.history['loss'])
plt.plot(mod_history.history['val_loss'])
plt.title('model MSE loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(np.argmin(mod_history.history['loss']),
      np.argmin(mod_history.history['val_loss']))
print(min(mod_history.history['loss']),
      min(mod_history.history['val_loss']))

best_mod = create_model()
best_mod.load_weights(os.path.join(WEIGHT_DIR, "mod6/cp.ckpt"))
best_mod_trainmse = best_mod.evaluate(X_train, y_train)
best_mod_testmse = best_mod.evaluate(X_test, y_test)
best_mod_trainrmse = best_mod_trainmse ** 0.5
best_mod_testrmse = best_mod_testmse ** 0.5
print(best_mod_trainmse, best_mod_testmse, best_mod_trainrmse, best_mod_testrmse)

# train and test mse & rmse for all models
models_loss=np.array([[30993.244140625, 29201.421875, 28928.923828125,
                          27136.6113, 18496.9453125, 13529.3379],
                     [37051.4375, 33454.6484375, 35293.98828125, 
                          28654.7812, 51757.32421875, 16877.2148],
                     [176.04898221979303, 170.88423530273352, 170.08504880830944,
                         164.7319377902324, 136.00347536919782, 116.3156820494339],
                     [192.48749959412947, 182.9061191909664, 187.86694302417868,
                         169.27723193034555, 227.50236090807937, 129.91233522552815]])

plt.plot(['model1', 'model2', 'model3', 'model4', 'model5', 'model6'], models_loss[2,])
plt.plot(['model1', 'model2', 'model3', 'model4', 'model5', 'model6'], models_loss[3,])
plt.title('Models RMSE Comparison')
plt.ylabel('RMSE')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
