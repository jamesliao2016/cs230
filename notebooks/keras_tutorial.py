
# https://machinelearningmastery.com/check-point-deep-learning-models-keras/

# In[7]:


# Checkpoint the weights when validation accuracy improves
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import numpy


# In[6]:


# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


# In[7]:



# create model
model = Sequential()
model.add(Dense(12,
                input_dim=8,
                kernel_initializer='uniform',
                bias_initializer='zeros',
                activation='relu'))

model.add(Dense(8,
                kernel_initializer='uniform',
                bias_initializer='zeros',
                activation='relu'))
model.add(Dense(1,
                kernel_initializer='uniform',
                bias_initializer='zeros',
                activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)


# In[ ]:



