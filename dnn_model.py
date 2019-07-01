import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, BatchNormalization, CuDNNLSTM
from keras import losses
from keras.optimizers import Adam,SGD
from keras.regularizers import l1, l2

#1538
def model_creator(leng=448, layer_size=512):
  #linear network
  model = Sequential()

  #model.add(Embedding(3+1, 8, input_length=leng))
  #model.add(Flatten())

  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu", input_dim=leng))

  model.add(BatchNormalization())
  #model.add(Dropout(0.2))  
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  #model.add(Dropout(0.2))
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  #model.add(Dropout(0.2))
  model.add(Dense(units=layer_size, kernel_initializer='normal', activation="relu"))
  model.add(Dense(units=1, activation="sigmoid"))

  #opt = SGD(lr=0.0001, momentum=0.9, nesterov=True) 
  opt = Adam(lr=0.001)
  model.compile(loss=losses.binary_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  print(model.summary())
  return model