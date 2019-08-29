# Databricks notebook source
from matplotlib import pyplot
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pyspark.sql.functions import *
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls

### Constants #####
figsize=(15, 7)

# Enable Arrow-based columnar data transfers to transfer Dataframe to Pandas
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

#Read data from table
ts_df = spark.sql("select * from KEY_CONTROLS").where("RESULT_KEY_NBR = 11").select('DATE','ACTL_VAL') \
    .withColumn('time',unix_timestamp(col('DATE'), "yyyy-MM-dd").cast("timestamp")) \
    .select('time','ACTL_VAL').orderBy('time')

#Convert to Pandas
init_notebook_mode(connected=True)
time_series_df=ts_df.toPandas()
time_series_df.d()
actual_vals = time_series_df.ACTL_VAL
actual_log = np.log10(actual_vals)



# COMMAND ----------

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(values=columns,axis=1)
    df.fillna(0, inplace=True)
    return df# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]#### LSTM
  

supervised = timeseries_to_supervised(actual_log, 1)
supervised_values = supervised.values# split data into train and test-sets
train_lstm, test_lstm = supervised_values[0:-70], supervised_values[-70:]# transform the scale of the data
scaler, train_scaled_lstm, test_scaled_lstm = scale(train_lstm, test_lstm)# fit the model                 batch,Epoch,Neurons
lstm_model = fit_lstm(train_scaled_lstm, 1, 850 , 3)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled_lstm[:, 0].reshape(len(train_scaled_lstm), 1, 1)
#lstm_model.predict(train_reshaped, batch_size=1)

# COMMAND ----------


