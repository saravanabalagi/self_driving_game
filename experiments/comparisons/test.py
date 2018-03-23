from preprocess import get_test_train_data
from model import accuracy, loss
from keras import backend
from save_load import load
import sys, os

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred = backend.cast(y_pred, 'float32')
    current_accuracy = accuracy(y_test, y_pred)
    current_loss = loss(y_test, y_pred)
    return [current_loss, current_accuracy]

if __name__ == '__main__':
    
    file = "F:\Projects\python\self_driving_game\data\dataset_mini.pz"
    if len(sys.argv)<=1: count = None
    else: count = int(sys.argv[1]);
    
    model = load(count)
    x_train, x_test, y_train, y_test = get_test_train_data(file, 1000, tanh=True)
    scores = evaluate_model(model, x_test, y_test)
    print("Loss: ", scores[0])
    print("Accuracy: ", scores[1]*100, "%")