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

def relu_with_scaled_sigmoid_model(channels=1, learning_rate=0.005, drop_rate=0.1, pick_last_steer=5):

    h = 75          # height of the image
    w = 100         # width of the image
    c = channels    # no of channels

    image = Input(shape=(h,w,c))
    conv_1a_1 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu', input_shape=(h, w, c), name='conv_1a_1')(image)
    conv_1a_2 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu', name='conv_1a_2')(conv_1a_1)
    pool_1a_2 = MaxPooling2D(pool_size=(2,2), name='pool_1a_2')(conv_1a_2)
    drop_1a_2 = Dropout(drop_rate, seed=0, name='drop_1a_2')(pool_1a_2)

    conv_2a_1 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2a_1')(drop_1a_2)
    conv_2a_2 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2a_2')(conv_2a_1)
    conv_2a_3 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2a_3')(conv_2a_2)
    pool_2a_3 = MaxPooling2D(pool_size=(2,2), name='pool_2a_3')(conv_2a_3)
    drop_2a_3 = Dropout(drop_rate, seed=0, name='drop_2a_3')(pool_2a_3)

    conv_3a_1 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3a_1')(drop_2a_3)
    conv_3a_2 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3a_2')(conv_3a_1)
    conv_3a_3 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3a_3')(conv_3a_2)
    pool_3a_3 = MaxPooling2D(pool_size=(2,2), name='pool_3a_3')(conv_3a_3)
    drop_3a_3 = Dropout(drop_rate, seed=0, name='drop_3a_3')(pool_3a_3)
    flatten_3a_3 = Flatten(name='flatten_3a_3')(drop_3a_3)

    minimap = Input(shape=(h,w,c))
    conv_1b_1 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu', input_shape=(h, w, c), name='conv_1b_1')(minimap)
    conv_1b_2 = Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu', name='conv_1b_2')(conv_1b_1)
    pool_1b_2 = MaxPooling2D(pool_size=(2,2), name='pool_1b_2')(conv_1b_2)
    drop_1b_2 = Dropout(drop_rate, seed=0, name='drop_1b_2')(pool_1b_2)

    conv_2b_1 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2b_1')(drop_1b_2)
    conv_2b_2 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2b_2')(conv_2b_1)
    conv_2b_3 = Conv2D(64, (3, 3), kernel_initializer='normal', activation='relu', name='conv_2b_3')(conv_2b_2)
    pool_2b_3 = MaxPooling2D(pool_size=(2,2), name='pool_2b_3')(conv_2b_3)
    drop_2b_3 = Dropout(drop_rate, seed=0, name='drop_2b_3')(pool_2b_3)

    conv_3b_1 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3b_1')(drop_2b_3)
    conv_3b_2 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3b_2')(conv_3b_1)
    conv_3b_3 = Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu', name='conv_3b_3')(conv_3b_2)
    pool_3b_3 = MaxPooling2D(pool_size=(2,2), name='pool_3b_3')(conv_3b_3)
    drop_3b_3 = Dropout(drop_rate, seed=0, name='drop_3b_3')(pool_3b_3)
    flatten_3b_3 = Flatten(name='flatten_3b_3')(drop_3b_3)
    
    previous_steer = Input(shape = (pick_last_steer,))
    dense_3c_1= Dense(40, kernel_initializer='normal', activation='relu')(previous_steer)
    
    speed_input = Input(shape=(1,))
    yaw_input = Input(shape=(1,))

    concat_4_0 = concatenate([flatten_3a_3, flatten_3b_3, dense_3c_1, speed_input, yaw_input])
    dense_4_1 = Dense(2048, kernel_initializer='normal', activation='relu', name='dense_4_1')(concat_4_0)
    dense_4_2 = Dense(1024, kernel_initializer='normal', activation='relu', name='dense_4_2')(dense_4_1)
    dense_4_3 = Dense(128, kernel_initializer='normal', activation='relu', name='dense_4_3')(dense_4_2)
    dense_4_4 = Dense(1, kernel_initializer='normal', activation='sigmoid', name='dense_4_4')(dense_4_3)
    output = ScaleLayer(name='output')(dense_4_4)

    model = Model(inputs=[image, minimap, previous_steer, speed_input, yaw_input], outputs=output)
    model.compile(loss=loss, optimizer=Adam(learning_rate), metrics=['mae',accuracy])
    return model
