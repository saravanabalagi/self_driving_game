from keras.callbacks import TensorBoard
from model import relu_model, tanh_model
from test import evaluate_model
from save_load import save, get_last_file_number
from preprocess import get_test_train_data
import tensorflow as tf
import sys, os

if __name__ == '__main__':

    # Print output to file
    outfolder = 'exp_' + '{0:03d}'.format(get_last_file_number(prefix='exp_', suffix='') + 1); os.makedirs(outfolder)
    outfile = outfolder + '/' + 'train_' + '{0:03d}'.format(get_last_file_number(path=outfolder) + 1) + '.log'
    print('Printing to logfile at', outfile)
    sys.stdout = open(outfile, 'w+')

    # Get test and train data
    file = "E:\Backup\GTA V Dataset\dataset_1.pz"
    x_train, x_test, y_train, y_test = get_test_train_data(file, 1000, tanh=True)
    # x_train, x_test, y_train, y_test = get_test_train_data(file, 1000, tanh=False)

    # Train model
    model = tanh_model()
    # model = relu_model()
    tbCallBack = TensorBoard(log_dir=outfolder, histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=250, callbacks=[tbCallBack])
    save(model, path=outfolder)

    # Evaluate model
    scores = evaluate_model(model, x_test, y_test)

    # Print scores
    print('\n\n')
    sess = tf.Session()
    print("Loss: ", sess.run(scores[0]))
    print("Accuracy: ", sess.run(scores[1])*100, "%")