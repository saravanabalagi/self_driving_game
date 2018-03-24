from preprocess import get_test_train_data
from save_load import get_last_file_number
from save_load import load
from model import accuracy, loss
from keras import backend
import sys, os
import tensorflow as tf
from utils import visualize_layers 

if __name__ == '__main__':
    
    file = "F:\Projects\python\self_driving_game\data\dataset_mini.pz"
    if len(sys.argv)<=1: count = None
    else: count = int(sys.argv[1]);
    
    # Load model
    exp_folder = 'exp_' + '{0:03d}'.format(get_last_file_number(prefix='exp_', suffix=''))
    model = load(count, path=exp_folder)

    # Visualize model
    x_train, x_test, y_train, y_test = get_test_train_data(file, 2, tanh=True)
    visualize_layers(model, x_test)
    print('Visualization done...!')