from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.metrics import mae
from keras.initializers import RandomNormal

from keras import backend
backend.set_image_dim_ordering('tf')

def loss(y_true, y_pred): return backend.sum(backend.abs(y_true - y_pred))
def accuracy(y_true, y_pred): return 1 - backend.mean(backend.minimum(1.0,((backend.abs(y_true - y_pred))*10)))

def tanh_model(channels=1):
    
    h = 75          # height of the image
    w = 100         # width of the image
    c = channels    # no of channels

    model = Sequential()
        
    model.add(Conv2D(32, (7, 7), kernel_initializer='normal', activation='tanh', input_shape=(h, w, c)))
    model.add(Conv2D(32, (7, 7), kernel_initializer='normal', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='tanh'))
    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='tanh'))
    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='tanh'))
    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='tanh'))
    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(2048, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1024, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(128, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=Adam(0.00001), metrics=['mae',accuracy])
    return model

def relu_model(channels=1):
    
    h = 75          # height of the image
    w = 100         # width of the image
    c = channels    # no of channels

    model = Sequential()
        
    model.add(Conv2D(32, (7, 7), kernel_initializer='normal', activation='relu', input_shape=(h, w, c)))
    model.add(Conv2D(32, (7, 7), kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='relu'))
    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='relu'))
    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu'))
    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu'))
    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=Adam(0.00001), metrics=['mae',accuracy])
    return model