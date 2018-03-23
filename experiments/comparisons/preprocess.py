import numpy as np
import argparse
import pickle
import gzip
import os
import cv2

def get_test_train_data(file, data_limit=-1, tanh=False):

    # Set seed
    np.random.seed(0)

    # Print Frame shape
    pfile = gzip.open(file, mode='rb')
    var = pickle.load(pfile)
    im_shape = var['frame'].shape
    print('Original Frame shape:', var['frame'].shape)

    # Load all variables
    count = 0
    images = []
    outputs = []
    im_reshape = (100, 75)
    pfile = gzip.open(file, mode='rb')
    while True:
        try:
            # Load var from pickle
            var = pickle.load(pfile)
            count += 1
                
            # Resize and load image
            image = var['frame']
            image = cv2.resize(image, im_reshape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[..., None]
            images.append(image)
            
            # Append outputs
            outputs.append(var['steering'])

            # Stopping criteria
            if data_limit!=-1 and count>=data_limit: break
        except EOFError: break

    x = np.array(images)
    y = np.array(outputs)

    print('Dataset Shape: x: {} | y: {}'.format(x.shape, y.shape))

    # Normalize data
    if tanh: x = (x/255 - 0.5) * 2
    else: x = x/255
    np.clip(y, -1, 1, out=y)

    # Test train split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random, shuffle=True)

    print()
    print("Train Data | Test Data")
    print(("{0:^10} | {1:^10}").format(y_train.shape[0], y_test.shape[0]))
    print()

    return x_train, x_test, y_train, y_test