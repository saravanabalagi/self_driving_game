from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.metrics import mae
from keras.initializers import RandomNormal

from keras import backend
backend.set_image_dim_ordering('tf')

def sequential(channels=1):
    
    h = 75          # height of the image
    w = 100         # width of the image
    c = channels    # no of channels

    model = Sequential()
        
    model.add(Conv2D(32, (7, 7), kernel_initializer='normal', activation='relu', input_shape=(h, w, c)))
    model.add(Conv2D(32, (7, 7), kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='relu'))
    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='relu'))
    model.add(Conv2D(64, (5, 5), kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu'))
    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu'))
    model.add(Conv2D(128, (3, 3), kernel_initializer='normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    # model.add(Conv2D(256, (3, 3), kernel_initializer='normal', activation='relu'))
    # model.add(Conv2D(256, (3, 3), kernel_initializer='normal', activation='relu'))
    # model.add(Conv2D(256, (3, 3), kernel_initializer='normal', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))
    # # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(2048, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='tanh'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

def image_steering_concat_model(channels=3):

    h = 75          # height of the image
    w = 100         # width of the image
    c = channels    # no of channels
    
    # Level 1
    image = Input(shape=(h,w,c))
    conv_1_1 = Conv2D(64, kernel_size=3, kernel_initializer='normal', activation='tanh', input_shape=(h,w,c))(image)
    conv_1_2 = Conv2D(64, kernel_size=3, kernel_initializer='normal', activation='tanh')(conv_1_1)
    pool_1_1 = MaxPooling2D(pool_size=(2, 2))(conv_1_2)
    
    # Level 2
    conv_2_1 = Conv2D(128, kernel_size=3, kernel_initializer='normal', activation='tanh')(pool_1_1)
    conv_2_2 = Conv2D(128, kernel_size=3, kernel_initializer='normal', activation='tanh')(conv_2_1)
    pool_2_1 = MaxPooling2D(pool_size=(2, 2))(conv_2_2)
    
    # Level 3a
    conv_3a_1 = Conv2D(256, kernel_size=3, kernel_initializer='normal', activation='tanh')(pool_2_1)
    conv_3a_2 = Conv2D(256, kernel_size=3, kernel_initializer='normal', activation='tanh')(conv_3a_1)
    conv_3a_3 = Conv2D(256, kernel_size=3, kernel_initializer='normal', activation='tanh')(conv_3a_2)
    pool_3a_1 = MaxPooling2D(pool_size=(2, 2))(conv_3a_3)
    pool_3a_1f = Flatten()(pool_3a_1)

    # Level 3b
    prev_steering = Input(shape=(pick_last_steering,))
    dense_3b_1 = Dense(40, kernel_initializer='normal', activation='tanh')(prev_steering)

    # Level 4
    concat_4_1 = concatenate([pool_3a_1f, dense_3b_1])
    dense_4_1 = Dense(4096, kernel_initializer='normal', activation='tanh')(concat_4_1)
    dense_4_2 = Dense(2048, kernel_initializer='normal', activation='tanh')(dense_4_1)
    dense_4_3 = Dense(1, kernel_initializer='normal', activation='tanh')(dense_4_2)

    model = Model(inputs=[image, prev_steering], outputs=dense_4_3)
    model.compile(loss='mse', optimizer=Adam(lr=0.00001), metrics=[mae,'accuracy'])
    return model