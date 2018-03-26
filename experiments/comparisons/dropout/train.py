from keras.callbacks import TensorBoard
from keras import backend
from model import relu_model, tanh_model, relu_with_scaled_sigmoid_model
from test import evaluate_model
from save_load import save, get_last_file_number
from preprocess import get_test_train_data
from utils import print_model_summary
import tensorflow as tf
import sys, os

if __name__ == '__main__':

    # # Print output to file
    # outfolder = 'exp_' + '{0:03d}'.format(get_last_file_number(prefix='exp_', suffix='') + 1); os.makedirs(outfolder)
    # outfile = outfolder + '/' + 'train_' + '{0:03d}'.format(get_last_file_number(path=outfolder) + 1) + '.log'
    # print('Printing to logfile at', outfile)
    # sys.stdout = open(outfile, 'w+')

    # Add title to logfile for identification
    # if len(sys.argv)>1: print('Title:',sys.argv[1],'\n\n')

    # Get test and train data
    file = os.environ['DATA_DIR']+ "/dataset_mini.pz"
    x_train, x_test, y_train, y_test = get_test_train_data(file, 2000, tanh=False)
    # x_train, x_test, y_train, y_test = get_test_train_data(file, 1000, tanh=False)

    learning_rates = 0.0005
    models = ['tanh', 'relu_with_scaled_sigmoid']
    drop_rates = [0.1, 0.2, 0.3, 0.4]

    for i in range(len(models)):
        for j in range(len(drop_rates)):

            # Print output to file
            outfolder = 'exp_' + '{0:03d}'.format(get_last_file_number(prefix='exp_', suffix='') + 1); os.makedirs(outfolder)
            outfile = outfolder + '/' + 'train_' + '{0:03d}'.format(get_last_file_number(path=outfolder) + 1) + '.log'
            print('Printing to logfile at', outfile)
            sys.stdout = open(outfile, 'w+')

            print('Title:', '{}_adam_{}_dropout_rate_{}'.format(models[i], learning_rate, drop_rates[j]),'\n\n')


            if models[i] == 'relu': model = relu_model(learning_rate=learning_rates[j])
            elif models[i] == 'relu_with_scaled_sigmoid': model = relu_with_scaled_sigmoid_model(learning_rate=learning_rates, drop_rate=drop_rates[i])
            elif models[i] == 'tanh': 
                model = tanh_model(learning_rate=learning_rates[j], drop_rate=drop_rates[i])
                x_train = (x_train - 0.5) * 2 
                x_test = (x_test - 0.5) * 2

            # Print model summary
            print_model_summary(model)

            # Train model
            tbCallBack = TensorBoard(log_dir=outfolder, histogram_freq=0, write_graph=True, write_images=True)
            model.fit(x_train, y_train, validation_split=0.2, epochs=25, batch_size=250, callbacks=[tbCallBack])
            save(model, path=outfolder)

            # Evaluate model
            scores = evaluate_model(model, x_test, y_test)

            # Print scores
            print('\n\n')
            print("Loss: ", backend.get_value(scores[0]))
            print("Accuracy: ", backend.get_value(scores[1])*100, "%")


    # # Create model and print to log file
    # model = tanh_model()
    # # model = relu_model()
    # print_model_summary(model)

    # # Train model
    # tbCallBack = TensorBoard(log_dir=outfolder, histogram_freq=0, write_graph=True, write_images=True)
    # model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=250, callbacks=[tbCallBack])
    # save(model, path=outfolder)

    # # Evaluate model
    # scores = evaluate_model(model, x_test, y_test)

    # # Print scores
    # print('\n\n')
    # print("Loss: ", backend.get_value(scores[0]))
    # print("Accuracy: ", backend.get_value(scores[1])*100, "%")
