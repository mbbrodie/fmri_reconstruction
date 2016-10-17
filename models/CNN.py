from __future__ import print_function
from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


def create_CNN():
    '''Create Simple Deep CNN model for Miyawaki fMRI scans based on
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py'''
    model = Sequential()
    model.add(Convolution2D(3, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(30, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(60, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(30, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(8000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.001, decay=1e-8, momentum=0.9, nesterov=True)
    #Consider passing these setting in by text file
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

class Miyawaki_CNN:
    '''CNN model for learning Miyawaki fMRI scans'''
    def init(self, saved_file=None):
        if saved_file is None:
            self.model = create_model()
        else:
            self.model = load_model(saved_file)

    def train():

    def predict():
