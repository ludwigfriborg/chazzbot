import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras import losses
from keras.optimizers import Adam

def model_creator():
  #linear network
  model = Sequential()

  model.add(Embedding(15, 8, input_length=128))
  model.add(Flatten())
  model.add(Dense(
    units=1024,
    kernel_initializer='normal',
    activation="relu",
    input_dim=131))
  model.add(Dropout(0.5))
  model.add(Dense(units=512, activation='relu', kernel_initializer='normal'))
  model.add(Dropout(0.2))
  model.add(Dense(units=512, activation='relu', kernel_initializer='normal'))
  model.add(Dropout(0.2))
  model.add(Dense(units=512, activation='relu', kernel_initializer='normal'))
  model.add(Dropout(0.2))
  model.add(Dense(units=512, activation='relu', kernel_initializer='normal'))
  model.add(Dropout(0.2))
  model.add(Dense(units=512, activation='relu', kernel_initializer='normal'))
  model.add(Dropout(0.2))
  model.add(Dense(units=256, activation='relu', kernel_initializer='normal'))
  model.add(Dropout(0.2))
  model.add(Dense(units=128, activation='relu', kernel_initializer='normal'))
  model.add(Dense(units=1, activation="sigmoid"))

  opt = Adam(lr=0.0001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model
