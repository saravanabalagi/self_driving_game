from keras.models import load_model
from model import loss, accuracy
import h5py
import os

def get_last_file_number(path):
    valid_model_files = ['.h5']
    numbers = [-1]
    for file in os.listdir(path):
        filename = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        if ext.lower() not in valid_model_files: continue
        if filename.startswith('model_'): 
            numbers.append(int(''.join(list(filter(str.isdigit, filename)))))
    counter = max(numbers)
    return counter

def save(model, path=None):
    counter = 0
    path = os.getcwd()
    counter = get_last_file_number(path) + 1

    model_name = 'model_' + '{0:03d}'.format(counter)
    model.save(path + '\\' + model_name + '.h5')

    print("Saving model:" + model_name + '.h5')

def load(count=None, path=None):
    if path is None: path = os.getcwd()
    if count is None: count = get_last_file_number(path)
    if count == -1: print("File not found"); return
    model_name = 'model_' + '{0:03d}'.format(count)
    model = load_model(path + '\\' + model_name + '.h5', custom_objects={'loss': loss, 'accuracy': accuracy})
    return model