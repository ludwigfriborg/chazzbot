import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras import losses
from keras.optimizers import Adam

def model_creator():
  #linear network
  model = Sequential()

  #iniializer
  init_s = keras.initializers.he_normal(seed=None)

  #network might work well without embeddings
  model.add(Embedding(15, 8, input_length=131))
  model.add(Flatten())
  model.add(Dense(
    units=248,
    kernel_initializer='normal',
    activation="relu",
    input_dim=131))
  model.add(Dropout(0.5))
  model.add(Dense(units=2048, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(units=2048, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(units=2048, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(units=512, activation='relu'))
  model.add(Dense(units=2, activation="sigmoid")) #sigmoid might not be the very best
  #keras.activations.softmax(x, axis=-1)

  opt = Adam(lr=0.001)
  model.compile(loss=losses.mean_absolute_error,
                optimizer=opt,
                metrics=['accuracy', 'mse'])
  return model
