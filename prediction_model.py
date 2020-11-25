import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def getTimeSeriesData(A, window=7):
  X, y = list(), list()
  for i in range(len(A)):
      end_ix = i + window
      if end_ix > len(A) - 1:
          break
      seq_x, seq_y = A[i:end_ix], A[end_ix]
      X.append(seq_x)
      y.append(seq_y)
  return np.array(X), np.array(y)

def returnRates(countryName):

  df = pd.read_csv("Foreign_Exchange_Rates.csv")

  df = df.drop(columns=["Unnamed: 0"])
  newColumnsNames = list(map(lambda c: c.split(" - ")[0] if "-" in c else "DATE", df.columns))
  df.columns = newColumnsNames

  df = df.replace("ND", np.nan)
  df = df.bfill().ffill() 

  df = df.set_index("DATE")
  df.index = pd.to_datetime(df.index)

  df = df.astype(float)

  num_records = (df.index.max()-df.index.min()).days+1

  data = {}
  data["DATE"] = pd.date_range("2000-01-03", df.index.max().strftime('%Y-%m-%d'), freq="D")

  complete = pd.DataFrame(data=data)
  complete = complete.set_index("DATE")
  complete = complete.merge(df, left_index=True, right_index=True, how="left")
  complete = complete.bfill().ffill()

  sampled2d = complete.resample("2D").mean()

  window = 2
  num_features = 1
  X, y = getTimeSeriesData(list(sampled2d[countryName]), window=window)
  X = X.reshape((X.shape[0], X.shape[1], num_features))  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  model = Sequential()
  model.add(LSTM(7, activation='relu', input_shape=(window, num_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  history = model.fit(X_train, y_train, epochs=20, verbose=1)

  yPred = model.predict(X_test, verbose=0)
  yPred.shape = yPred.shape[0]

  days = 5
  result = list()
  yPred1 = model.predict(X, verbose=0)
  result.append(yPred1[-25:])
  x1 = X

  for i in range(days):
    z = list()
    ar = list()
    val = X[-1:]
    ar.append(val[0,1])
    ar.append(yPred1[-1:])
    z = np.array(ar)
    w = np.append(X,z)
    w = w.reshape(w.size//2,2,1)
    X = np.asarray(w).astype(np.float32)
    yPred1 = model.predict(X, verbose=0)
    result.append(yPred1[-1:])

  rates = []
  for val in range(25):
    rates.append(round(float(result[0][val]), 4))

  for val in range(1, 6):
    rates.append(round(float(result[val]), 4))

  return rates