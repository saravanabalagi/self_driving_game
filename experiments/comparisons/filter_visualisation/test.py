from preprocess import get_test_train_data
from save_load import get_last_file_number
from save_load import load
from model import accuracy, loss
from keras import backend
import sys, os
import tensorflow as tf

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = backend.cast(y_pred, 'float32')
    current_accuracy = accuracy(y_test, y_pred)
    current_loss = loss(y_test, y_pred)
    return [current_loss, current_accuracy]

if __name__ == '__main__':
    
    # file = "F:\Projects\python\self_driving_game\data\dataset_mini.pz"
    file = os.environ['DATA_DIR']+ "/dataset_75p_gray.pz"
    if len(sys.argv)<=1: count = None
    else: count = int(sys.argv[1]);
    
    # Load model
    exp_folder = 'experiments/comparison/dropout/exp_' + '{0:03d}'.format(get_last_file_number(prefix='exp_', suffix=''))
    model = load(count, path=exp_folder)

    # Evaluate model
    x_train, x_test, y_train, y_test = get_test_train_data(file, 10000, tanh=True)
    print('x_test shape', x_test.shape)
    scores = evaluate_model(model, x_test, y_test)

    # Print scores
    print('\n\n')
    print("Loss: ", backend.get_value(scores[0]))
    print("Accuracy: ", backend.get_value(scores[1])*100, "%")
