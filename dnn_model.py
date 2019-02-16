import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras import losses
from keras.optimizers import Adam

def model_creator():
  #linear network
  model = Sequential()

  #network might work well without embeddings
  model.add(Embedding(15, 8, input_length=128))
  model.add(Flatten())
  model.add(Dense(
    units=512,
    kernel_initializer='normal',
    activation="relu",
    input_dim=131))
  model.add(Dropout(0.5))
  model.add(Dense(units=1024, activation='relu', kernel_initializer='normal'))
  model.add(Dropout(0.2))
  model.add(Dense(units=1024, activation='relu', kernel_initializer='normal'))
  model.add(Dropout(0.2))
  model.add(Dense(units=512, activation='relu', kernel_initializer='normal'))
  model.add(Dense(units=1, activation="sigmoid")) #sigmoid might not be the very best
  #keras.activations.softmax(x, axis=-1)

  opt = Adam(lr=0.001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])
  return model
