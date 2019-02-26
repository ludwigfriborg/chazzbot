import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, BatchNormalization, LeakyReLU
from keras import losses
from keras.optimizers import Adam

def model_creator():
  #linear network
  model = Sequential()

  model.add(Embedding(15, 8, input_length=128))
  model.add(Flatten())
  model.add(BatchNormalization())
  model.add(Dense(
    units=512,
    kernel_initializer='normal',
    input_dim=131))
  model.add(LeakyReLU())
  model.add(Dropout(0.5))
  model.add(Dense(units=512, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dropout(0.2))
  model.add(Dense(units=512, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dropout(0.2))
  model.add(Dense(units=512, kernel_initializer='normal'))
  model.add(LeakyReLU())

  model.add(Dense(units=256, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=256, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=128, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=64, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=1, activation="sigmoid"))

  opt = Adam(lr=0.0001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model


from keras.layers import Conv2D, MaxPooling2D
def model_creator_cnn():
  #linear network
  model = Sequential()

  # model.add(Embedding(15, 64, input_length=128))
  model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(16, 8, 1)))
  model.add(BatchNormalization())
  model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(MaxPooling2D((2,2)))
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
  model.add(Flatten())

  model.add(Dense(units=128, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=64, kernel_initializer='normal'))
  model.add(LeakyReLU())
  model.add(Dense(units=1, activation="sigmoid"))

  opt = Adam(lr=0.0001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model
