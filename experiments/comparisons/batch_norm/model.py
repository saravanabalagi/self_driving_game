from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.metrics import mae
from keras.initializers import RandomNormal
from keras.engine.topology import Layer
import numpy as np

from keras import backend
backend.set_image_dim_ordering('tf')

class ScaleLayer(Layer):
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
    def call(self, x):
        return (x-0.5)*2
    def compute_output_shape(self, input_shape):
        return input_shape
    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

def loss(y_true, y_pred):     return backend.sum(backend.abs(y_true - y_pred))
def accuracy(y_true, y_pred): return 1 - backend.mean(backend.minimum(1.0,((backend.abs(y_true - y_pred))*10)))

loss.__name__ == 'Sum of Absolute Errors'
accuracy.__name__ == 'Mean of Absolute Errors of upto 0.1'

def tanh_model(channels=1, learning_rate=0.001, drop_rate=0.2):
    
    h = 75          # height of the image
    w = 100         # width of the image
    c = channels    # no of channels

    image = Input(shape=(h,w,c))      
    conv_1_1 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='tanh', input_shape=(h, w, c), name='conv_1_1')(image)
    conv_1_2 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='tanh', name='conv_1_2')(conv_1_1)
    pool_1_2 = MaxPooling2D(pool_size=(2,2), name='pool_1_2')(conv_1_2)
    bnorm_1_2 = BatchNormalization()(pool_1_2)

    conv_2_1 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='tanh', name='conv_2_1')(bnorm_1_2)
    conv_2_2 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='tanh', name='conv_2_2')(conv_2_1)
    conv_2_3 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='tanh', name='conv_2_3')(conv_2_2)
    pool_2_3 = MaxPooling2D(pool_size=(2,2), name='pool_2_3')(conv_2_3)
    bnorm_2_3 = BatchNormalization()(pool_2_3)

	
    conv_3_1 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='tanh', name='conv_3_1')(bnorm_2_3)
    conv_3_2 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='tanh', name='conv_3_2')(conv_3_1)
    conv_3_3 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='tanh', name='conv_3_3')(conv_3_2)
    pool_3_3 = MaxPooling2D(pool_size=(2,2), name='pool_3_3')(conv_3_3)
    bnorm_3_3 = BatchNormalization()(pool_3_3)
    flatten_3_3 = Flatten(name='flatten_3_3')(bnorm_3_3)

    dense_4_1 = Dense(2048, kernel_initializer='normal', activation='tanh', name='dense_4_1')(flatten_3_3)
    dense_4_2 = Dense(1024, kernel_initializer='normal', activation='tanh', name='dense_4_2')(dense_4_1)
    dense_4_3 = Dense(128, kernel_initializer='normal', activation='tanh', name='dense_4_3')(dense_4_2)
    output = Dense(1, kernel_initializer='normal', activation='tanh', name='output')(dense_4_3)

    model = Model(inputs=image, outputs=output)
    model.compile(loss=loss, optimizer=Adam(learning_rate), metrics=['mae',accuracy])
    return model

def relu_model(channels=1, learning_rate=0.001):
    
    h = 75          # height of the image
    w = 100         # width of the image
    c = channels    # no of channels

    image = Input(shape=(h,w,c))      
    conv_1_1 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu', input_shape=(h, w, c), name='conv_1_1')(image)
    conv_1_2 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu', name='conv_1_2')(conv_1_1)
    pool_1_2 = MaxPooling2D(pool_size=(2,2), name='pool_1_2')(conv_1_2)

    conv_2_1 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2_1')(pool_1_2)
    conv_2_2 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2_2')(conv_2_1)
    conv_2_3 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2_3')(conv_2_2)
    pool_2_3 = MaxPooling2D(pool_size=(2,2), name='pool_2_3')(conv_2_3)

    conv_3_1 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3_1')(pool_2_3)
    conv_3_2 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3_2')(conv_3_1)
    conv_3_3 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3_3')(conv_3_2)
    pool_3_3 = MaxPooling2D(pool_size=(2,2), name='pool_3_3')(conv_3_3)
    flatten_3_3 = Flatten(name='flatten_3_3')(pool_3_3)

    dense_4_1 = Dense(2048, kernel_initializer='normal', activation='relu', name='dense_4_1')(flatten_3_3)
    dense_4_2 = Dense(1024, kernel_initializer='normal', activation='relu', name='dense_4_2')(dense_4_1)
    dense_4_3 = Dense(128, kernel_initializer='normal', activation='relu', name='dense_4_3')(dense_4_2)
    dense_4_4 = Dense(1, kernel_initializer='normal', activation='sigmoid', name='dense_4_4')(dense_4_3)
    output = ScaleLayer(1, name='output')(dense_4_4)
    model = Model(inputs=image, outputs=output)
    model.compile(loss=loss, optimizer=Adam(learning_rate), metrics=['mae',accuracy])
    return model


def relu_with_scaled_sigmoid_model(channels=1, learning_rate=0.001):

    h = 75          # height of the image
    w = 100         # width of the image
    c = channels    # no of channels

    image = Input(shape=(h,w,c))
    conv_1_1 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu', input_shape=(h, w, c), name='conv_1_1')(image)
    conv_1_2 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu', name='conv_1_2')(conv_1_1)
    pool_1_2 = MaxPooling2D(pool_size=(2,2), name='pool_1_2')(conv_1_2)
    bnorm_1_2 = BatchNormalization()(pool_1_2)

    conv_2_1 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2_1')(bnorm_1_2)
    conv_2_2 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2_2')(conv_2_1)
    conv_2_3 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2_3')(conv_2_2)
    pool_2_3 = MaxPooling2D(pool_size=(2,2), name='pool_2_3')(conv_2_3)
    bnorm_2_3 = BatchNormalization()(pool_2_3)

    conv_3_1 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3_1')(bnorm_2_3)
    conv_3_2 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3_2')(conv_3_1)
    conv_3_3 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3_3')(conv_3_2)
    pool_3_3 = MaxPooling2D(pool_size=(2,2), name='pool_3_3')(conv_3_3)
    bnorm_3_3 = BatchNormalization()(pool_3_3)
    flatten_3_3 = Flatten(name='flatten_3_3')(bnorm_3_3)

    dense_4_1 = Dense(2048, kernel_initializer='normal', activation='relu', name='dense_4_1')(flatten_3_3)
    dense_4_2 = Dense(1024, kernel_initializer='normal', activation='relu', name='dense_4_2')(dense_4_1)
    dense_4_3 = Dense(128, kernel_initializer='normal', activation='relu', name='dense_4_3')(dense_4_2)
    dense_4_4 = Dense(1, kernel_initializer='normal', activation='sigmoid', name='dense_4_4')(dense_4_3)
    output = ScaleLayer(name='output')(dense_4_4)

    model = Model(inputs=image, outputs=output)
    model.compile(loss=loss, optimizer=Adam(learning_rate), metrics=['mae',accuracy])
    return model
