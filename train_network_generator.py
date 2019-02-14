import keras
import numpy as np
import math
import json
import os
from random import shuffle

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
def get_training_data(batch_size):
  print("loading files...")
  count = 0

  #find all files
  file_names = os.listdir("./ext")
  shuffle(file_names)

  pop_new = True
  while len(file_names):
    if pop_new:
      file_name = file_names.pop()
      count = 0
      print('   ' + file_name, end='\r')
      pop_new = False

    filename, file_extension = os.path.splitext(file_name)
    if file_extension != '.json':
      pop_new = True
      continue

    #splice in data here not only latest file
    x, y = return_training_data(file_name, batch_size, count)
    if len(x) < batch_size:
      pop_new = True
      continue
    yield x, y

    count += batch_size
  print('woh')

def return_training_data(file_name, batch_size, point):
    with open("ext/" + file_name, "r") as file:
      X = []
      Y = []
      data = json.load(file)
      for x in data[point:point+batch_size]:
        tmp = x['board'] + x['move']
        X.append(tmp)
        Y.append([x['winning']])

      return np.array(X), np.array(Y)

# should add training function and so on
def train_network(model_name):
  epochs = 10
  batch_size = 750
  samples_per_epoch = 200
  validation_steps = 100
  evaluate_samples_per_epoch = 200

  model_filepath = "model/" + model_name + ".h5"

  tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph/' + model_name, histogram_freq=0, write_graph=True, write_images=True)
  checkpointCallBack=ModelCheckpoint(model_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

  model = model_creator()
  model.fit_generator(get_training_data(batch_size), epochs=epochs, samples_per_epoch=samples_per_epoch, callbacks=[checkpointCallBack], validation_data=get_training_data(batch_size), validation_steps=validation_steps)
  loss_and_metrics = model.evaluate_generator(get_training_data(batch_size), steps=evaluate_samples_per_epoch, verbose=0)
  print(loss_and_metrics)
  print(model.metrics_names[1] + ": " + str(loss_and_metrics[1] * 100))

  model.save(model_filepath)
