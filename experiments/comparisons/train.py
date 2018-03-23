from keras.callbacks import TensorBoard
from model import relu_model, tanh_model
from test import evaluate_model
from save_load import save
from preprocess import get_test_train_data
import sys

# Print output to file
sys.stdout = open('log_tanh', 'w+')

if __name__ == '__main__':
    file = "F:\Projects\python\self_driving_game\data\dataset_mini.pz"
    x_train, x_test, y_train, y_test = get_test_train_data(file, 1000, tanh=True)
    # x_train, x_test, y_train, y_test = get_test_train_data(file, 1000, tanh=False)

    model = tanh_model()
    # model = relu_model()
    tbCallBack = TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(x_train, y_train, validation_split=0.1, epochs=25, batch_size=125, callbacks=[tbCallBack])
    save(model)

    # Evaluate model
    scores = evaluate_model(model, x_test, y_test)

    # Print scores
    print('\n\n')
    sess = tf.Session()
    print("Loss: ", sess.run(scores[0]))
    print("Accuracy: ", sess.run(scores[1])*100, "%")