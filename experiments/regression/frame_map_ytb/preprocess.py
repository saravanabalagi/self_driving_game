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
    minimaps = []
    speeds = []
    yawRates = []
    pfile = gzip.open(file, mode='rb')
    while True:
        try:
            # Load var from pickle
            var = pickle.load(pfile)
            count += 1
                
            # Resize and load image
            image = var['frame']
            if(len(image.shape)==2):
                image = image[..., None]
            images.append(image)
            
            minimap = var['minimap']
            if(len(minimap.shape)==2):
                minimap = minimap[..., None]
            minimaps.append(minimap)
            
            speeds.append(var['speed'])
            yawRates.append(var['yawRate'])
            
            # Append outputs
            outputs.append([var['steering']])

            # Stopping criteria
            if data_limit!=-1 and count>=data_limit: break
        except Exception: break
    
    images = np.array(images)
    minimaps = np.array(minimaps)
    speeds = np.array(speeds)
    yawRates = np.array(yawRates)
    x = [images, minimaps, speeds, yawRates]
    y = np.array(outputs)

    x_shape = [entity.shape for entity in x]
    print('Dataset Shape: x: {} | y: {}'.format(x_shape, y.shape))

    # Normalize data
    if not isinstance(x, np.ndarray): 
    	x_new = [(entity/255) for i, entity in enumerate(x) if i<2]
    	if len(x)>=2: x_new = [*x_new, *x[2:]]
    	x = x_new
    else: x=x/255
    np.clip(y, -1, 1, out=y)

    # Test train split
    from sklearn.model_selection import train_test_split
    *recv,  y_train, y_test = train_test_split(*x, y, test_size=0.2, random_state=np.random, shuffle=True)
    
    x_train = recv[::2]
    x_test = recv[1::2]

    print()
    print("Train Data | Test Data")
    print(("{0:^10} | {1:^10}").format(y_train.shape[0], y_test.shape[0]))
    print()

    return x_train, x_test, y_train, y_test