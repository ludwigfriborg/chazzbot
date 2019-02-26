import keras
import numpy as np
import math
import json
import os
from random import shuffle

from dnn_model import model_creator, model_creator_cnn
from data_extractor import reshape_moves

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
  data = {}

  #find all files
  file_names = os.listdir("./ext")
  shuffle(file_names)

  pop_new = True
  while len(file_names):
    if pop_new:
      file_name = file_names.pop()
      file_extension = os.path.splitext(file_name)[1]
      if file_extension != '.json':
        pop_new = True
        continue
      count = 0
      print('   ' + file_name, end='\r')
      pop_new = False

      with open("ext/" + file_name, "r") as file:
        data = json.load(file)

    #splice in data here not only latest file
    x, y = return_training_data(batch_size, count, data)
    if len(x) < batch_size:
      pop_new = True
      continue
    yield x, y

    count += batch_size

    if not len(file_names):
      file_names = os.listdir("./ext")
      shuffle(file_names)
  print('woh')

def return_training_data(batch_size, point, data):
  X = []
  Y = []
  for x in data[point:point+batch_size]:
    tmp = reshape_moves(x['board'], x['move'])
    X.append(tmp)
    Y.append([x['winning']])

  return np.array(X), np.array(Y)

# should add training function and so on
def train_network(model_name):
  # Data set total size: ~16 000 000, (now ~32 000 000)
  epochs = 25
  batch_size = 256
  samples_per_epoch = 125000 # 125 000 for one epoch
  validation_steps = 200
  evaluate_samples_per_epoch = 100
  logging_freq = 50000 # number of samples

  model_filepath = "model/" + model_name + ".h5"

  tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph/' + model_name, histogram_freq=0, write_graph=True, write_images=True, update_freq=logging_freq)
  checkpointCallBack=ModelCheckpoint(model_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

  try:
    model = load_model(model_filepath)
    print('Loaded prevoisly saved model')
  except:
    model = model_creator()
    print('Created new model')


  model.fit_generator(get_training_data(batch_size), epochs=epochs, steps_per_epoch=samples_per_epoch, callbacks=[checkpointCallBack, tbCallBack], validation_data=get_training_data(batch_size), validation_steps=validation_steps)
  loss_and_metrics = model.evaluate_generator(get_training_data(batch_size), steps=evaluate_samples_per_epoch, verbose=0)
  print(loss_and_metrics)
  print(model.metrics_names[1] + ": " + str(loss_and_metrics[1] * 100))

  model.save(model_filepath)

def evaluate_model(model):
  evaluate_samples_per_epoch = 100
  batch_size = 256
  loss_and_metrics = model.evaluate_generator(get_training_data(batch_size), steps=evaluate_samples_per_epoch, verbose=0)
  print(loss_and_metrics)
  print(model.metrics_names[1] + ": " + str(loss_and_metrics[1] * 100))