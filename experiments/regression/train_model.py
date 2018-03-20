from tqdm import tqdm
import numpy as np
import argparse
import pickle
import gzip
import os
import cv2

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-d', '--dataset_path', default='../data/dataset.pz', help='Place to store the dataset')
    parser.add_argument('-dl', '--data_limit', default=-1, help='How much data to look for in the dataset? -1 for no limit')
    parser.add_argument('-m', '--maps', default=0, help='0 Dont use; 1 Concat with image; 2 Use Separately')
    parser.add_argument('-t', '--throttle', default=False, help='Use throttle or not?')
    parser.add_argument('-b', '--brake', default=False, help='Use brake or not?')
    parser.add_argument('-p', '--previous_data', default=0, help='How much previous_data to use')
    parser.add_argument('-r', '--random_seed', default=None, help='Seed for debugging')
    parser.add_argument('-e', '--epochs', default=None, help='How many epochs to train?')
    parser.add_argument('-e', '--learning_rate', default=None, help='What learning rate to set?')
    args = parser.parse_args()

    # Set random seed
    if args.random_seed is not None: np.random.seed(args.random_seed)

    # Read one set to get all params
    pfile = gzip.open(args.dataset_path, mode='rb')
    var = pickle.load(pfile)
    assert 'frame' in var.keys(), 'Could not find dict["frame"]';      im = var['frame']
    assert len(im.shape) == 3, 'Frame has more than 3 layers?';        im_shape  = (im.shape[1], im.shape[0])
    if args.maps!=0:
        assert 'map' in var.keys(), 'Could not find dict["map"]';          mp = var['map']
        assert len(mp.shape) == 3, 'Frame has more than 3 layers?';        mp_shape  = (mp.shape[1], mp.shape[0])

    # Find resize ratio
    diff = []
    diff.append(mp_shape[0] - im_shape[0])
    diff.append(mp_shape[1] - im_shape[1])
    ratio = 1- (np.min(diff) / mp_shape[np.argmin(diff)])
    to_subt = (np.round(np.array(mp_shape) * ratio) - im_shape).astype(int)

    # Load all variables
    images = []
    outputs = []
    if args.maps!=0: 
        minimaps = []
    pfile = gzip.open(args.dataset_path, mode='rb')
    while True:
        try:
            # Load var from pickle
            var = pickle.load(pfile)
            count += 1
                
            # Resize and load image
            image = var['frame']
            image = cv2.resize(image, im_shape)
            images.append(image)
            
            # Crop maps to same size
            if args.maps != 0:
                minimap = var['map']
                minimap = cv2.resize(minimap, (0,0), fx=ratio, fy=ratio)
                minimap = minimap[to_subt[1]:,:minimap.shape[1]-to_subt[0],:]
                minimaps.append(minimap)
            
            # Append outputs
            current_output = []
            current_output.append(var['steering'])
            if args.throttle: current_output.append(var['throttle'])
            if args.brake:    current_output.append(var['brake'])
            outputs.append(current_output)

            # Stopping criteria
            if args.data_limit!=-1 and count>args.data_limit: break
        except EOFError: break

    # Print shape of dataset
    images = np.array(images)
    minimaps = np.array(minimaps)

    # Get inputs and outputs
    if args.maps==1: x = np.concatenate((images, minimaps), axis=1)
    else if args.maps==2: x = [images, minimaps]
    y = np.array(outputs)

    if not isinstance(x, np.ndarray): 
        x_shape = [entity.shape for entity in x]
    else: x_shape = x.shape
    print('Dataset Shape: x: {} | y: {}'.format(x_shape, y.shape))

    # Normalize data
    if not isinstance(x, np.ndarray): 
        x_new = [(entity/255 - 0.5) * 2 for i, entity in enumerate(x) if i<2]
        if len(x)>=2: x_new = [*x_new, *x[2:]]
        x = x_new
    else: x = (x/255 - 0.5) * 2
    np.clip(y, -1, 1, out=y)

    # Test train split
    from sklearn.model_selection import train_test_split
    if not isinstance(a, np.ndarray):
        recv = [None for i in range(2*len(x))]
        *recv, y_train, y_test = train_test_split(*x, y, test_size=0.2, random_state=np.random, shuffle=True)
        x_train = recv[::2]
        x_test = recv[1::2]
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random, shuffle=True)

    print("Train Data | Test Data")
    print(("{0:^10} | {1:^10}").format(y_train.shape[0], y_test.shape[0]))
