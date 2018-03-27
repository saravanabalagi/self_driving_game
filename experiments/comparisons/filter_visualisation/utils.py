from keras import backend
from keras.models import Model
import numpy as np
import cv2
import os

def print_optimizer(model):
	print()
	print('{:>20} : {}'.format('Optimizer', type(model.optimizer).__name__))
	print('_________________________________________________________________\n')
	print('{:>20} : {:7.6f}'.format('lr', backend.get_value(model.optimizer.lr)))
	print('{:>20} : {:7.6f}'.format('beta_1', backend.get_value(model.optimizer.beta_1)))
	print('{:>20} : {:7.6f}'.format('beta_2', backend.get_value(model.optimizer.beta_2)))
	print('{:>20} : {}'.format('decay', backend.get_value(model.optimizer.decay)))
	print('{:>20} : {}'.format('amsgrad', model.optimizer.amsgrad))
	print('{:>20} : {}'.format('epsilon', model.optimizer.epsilon))
	print('{:>20} : {}'.format('initial_decay', model.optimizer.initial_decay))
	print('_________________________________________________________________\n')

def print_metrics(model):
	print('{:>20} : {}'.format('Loss Function', model.loss if not callable(model.loss) else model.loss.__name__))
	print('_________________________________________________________________\n')
	for i,metric in enumerate(model.metrics):
		print('{:>18} {} : {}'.format('Metric',i, metric if not callable(metric) else metric.__name__))
	print('_________________________________________________________________\n')

def print_model_summary(model):
	model.summary()
	print_optimizer(model)
	print_metrics(model)

def denormalize(normalized_image):
	return ((normalized_image/2 + 0.5) * 255).astype(np.uint8)

def visualize_layers(model, test_input, path=None):
	if path is None: path = os.getcwd()
	for layer in model.layers:
		intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer.name).output)
		intermediate_output = intermediate_layer_model.predict(test_input)[-1]
		intermediate_output = denormalize(intermediate_output)
		if len(intermediate_output.shape) != 3: continue
		else: intermediate_output = np.swapaxes(intermediate_output,0,2)
		for index, image in enumerate(intermediate_output):
			filename = '{}_{:04d}{}'.format(layer.name, index, '.jpg')
			cv2.imwrite(path + '/' + filename, image)
