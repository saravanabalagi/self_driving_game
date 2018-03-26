from keras.models import load_model
from model import loss, accuracy
from model import ScaleLayer
import h5py
import os

def get_last_file_number(path=None, prefix='model_', suffix='.h5'):
    if path is None: path = os.getcwd()
    valid_model_files = [suffix]
    numbers = [-1]
    for file in os.listdir(path):
        filename = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        if ext.lower() not in valid_model_files: continue
        if filename.startswith(prefix): 
            numbers.append(int(''.join(list(filter(str.isdigit, filename)))))
    count = max(numbers)
    return count

def save(model, path=None, count=None):
    if path is None: path = os.getcwd()
    if count is None: count = get_last_file_number(path) + 1

    model_name = 'model_' + '{0:03d}'.format(count)
    model.save(path + '/' + model_name + '.h5')

    print("Saving model:" + model_name + '.h5')

def load(count=None, path=None):
    if path is None: path = os.getcwd()
    if count is None: count = get_last_file_number(path)
    if count == -1: print("File not found"); return
    model_name = 'model_' + '{0:03d}'.format(count)
    model = load_model(path + '\\' + model_name + '.h5', custom_objects={'loss': loss, 'accuracy': accuracy, 'ScaleLayer': ScaleLayer})
    print('Loaded model:',path + '\\' + model_name + '.h5')
    return model
