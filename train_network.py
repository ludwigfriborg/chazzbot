import keras
import numpy as np
import math
import json
import os

from dnn_model import model_creator

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint

#-----------------------------------------
# get trining data
#-----------------------------------------

#get all data from files
#need to use GENERATOR
def get_training_data(number_of_files):
  boards = np.empty([0, 896])
  win = np.empty([0, 1])

  print("loading files...")
  count = 0

  #find all files
  for file_name in os.listdir("./ext"):
    filename, file_extension = os.path.splitext(file_name)

    count += 1

    if file_extension != '.json':
      continue

    if count > number_of_files:
      break

    print(str(count) + "/" + str(number_of_files) + ": " + file_name)
    #splice in data here not only latest file
    boards_n, win_n = return_training_data(file_name)
    boards = np.vstack((boards, boards_n))
    win = np.vstack((win, win_n))

  print("Found " + str(count) + " different files with chess data")
  return boards, win

def return_training_data(file_name):
    with open("ext/" + file_name, "r") as file:
      X = []
      Y = []
      data = json.load(file)
      for x in data:
        X.append(x[:-1])
        Y.append([x[-1]])

      return np.array(X), np.array(Y)

def home_made_train_test_split(x, y, test_size=0.25):
  size = math.floor(x.shape[0] * test_size)

  x_train = x[size:]
  y_train = y[size:]
  x_test = x[:size]
  y_test = y[:size]

  return x_train, x_test, y_train, y_test

# should add training function and so on
def train_network(model_name):
  epochs = 2
  batch_size = 128
  number_of_files = 22

  X, Y = get_training_data(number_of_files)
  model_filepath = "model/" + model_name + ".h5"

  print(X.shape)

  #board_train, board_test, move_train, move_test = train_test_split(boards, moves, test_size=0.25)
  x_train, x_test, y_train, y_test = home_made_train_test_split(X, Y, test_size=0.125)

  tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph/' + model_name, histogram_freq=0, write_graph=True, write_images=True)
  checkpointCallBack=ModelCheckpoint(model_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  #esCallBack = EarlyStopping(monitor='val_acc', patience=5, min_delta=0.0001)

  model = model_creator()
  model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tbCallBack, checkpointCallBack], validation_data=(x_test, y_test), shuffle=True)
  loss_and_metrics = model.evaluate(x_test, y_test, verbose=0)
  print(loss_and_metrics)
  print(model.metrics_names[1] + ": " + str(loss_and_metrics[1] * 100))

  model.save(model_filepath)

def evaluate_model(model):
  number_of_files = 6

  X, Y = get_training_data(number_of_files)
  loss_and_metrics = model.evaluate(X, Y, verbose=0)
  print('Evaluated:', len(X), 'files')
  print(model.metrics_names[1] + ": " + str(loss_and_metrics[1] * 100))