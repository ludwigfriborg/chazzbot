import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, BatchNormalization, LeakyReLU
from keras import losses
from keras.optimizers import Adam,SGD
from keras.regularizers import l1, l2

def model_creator():
  #linear network
  model = Sequential()

  layer_size = 1538
  model.add(Dense(
    units=layer_size,
    kernel_initializer='normal',
    activation="relu",
    input_dim=896))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))  
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.2))
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  model.add(BatchNormalization())
  model.add(Dense(units=1, activation="sigmoid"))

  #opt = SGD(lr=0.0001, momentum=0.9, nesterov=True) 
  opt = Adam(lr=0.0001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model