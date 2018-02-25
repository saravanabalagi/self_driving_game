from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import regularizers
import sys, os

from keras import backend
backend.set_image_dim_ordering('th')
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
PATH_MODELS = os.path.join(os.getcwd(), '../models')

c = 1
h = 75
w = 100
no_of_classes = 9

model = Sequential()
    
model.add(Conv2D(32, (7, 7), kernel_initializer='normal', activation='relu', input_shape=(c, h, w)))
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
model.add(Dropout(0.5))
model.add(Dense(no_of_classes, kernel_initializer='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

counter = int(sys.argv[1])
model_name = 'model_' + '{0:03d}'.format(counter) + '.h5'
print("Saving", model_name, end='\r')
model.save(PATH_MODELS + '\\' + model_name)
print(model_name, "saved successfully")
