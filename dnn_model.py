import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, BatchNormalization, LeakyReLU
from keras import losses
from keras.optimizers import Adam

def model_creator():
  #linear network
  model = Sequential()

  model.add(Embedding(3, 8, input_length=896)) #896
  model.add(Flatten())
  model.add(BatchNormalization())
  model.add(Dense(
    units=1024,
    kernel_initializer='normal',
    activation='linear',
    input_dim=896))
  model.add(Dense(units=1024, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dropout(0.2))
  model.add(Dense(units=1024, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=512, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=512, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=1, activation="sigmoid"))

  opt = Adam(lr=0.0001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model