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
  bias_init = keras.initializers.Ones()

  #network might work well without embeddings
  model.add(Embedding(15, 8, input_length=131))
  model.add(Flatten())
  model.add(Dense(
    units=248,
    kernel_initializer='normal',
    bias_initializer=bias_init,
    activation="relu",
    input_dim=131))
  model.add(Dropout(0.4))
  model.add(Dense(units=2, activation="sigmoid")) #sigmoid might not be the very best

  opt = Adam(lr=0.001)
  model.compile(loss=losses.mean_squared_error,
                optimizer=opt,
                metrics=['accuracy', 'mse'])
  return model
