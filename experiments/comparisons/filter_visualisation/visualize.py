from preprocess import get_test_train_data
from save_load import get_last_file_number
from save_load import load
from model import accuracy, loss
from keras import backend
import sys, os
import tensorflow as tf
from utils import visualize_layers 

if __name__ == '__main__':
    
    file = os.environ['DATA_DIR']+ "/dataset_mini.pz"
    if len(sys.argv)<=1: count = None
    else: count = int(sys.argv[1]);
    
    # Load model
    exp_folder = 'exp_' + '{0:03d}'.format(get_last_file_number(prefix='exp_', suffix=''))
    model = load(count, path=exp_folder)

    # Visualize model
    x_train, x_test, y_train, y_test = get_test_train_data(file, 2, tanh=False)
    visualization_folder = exp_folder + '/visualization'
    if not os.path.exists(visualization_folder): os.makedirs(visualization_folder)

    visualize_layers(model, x_test, path=visualization_folder)
    print('Visualization done...!')
