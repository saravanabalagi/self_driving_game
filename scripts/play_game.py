from keras.models import load_model
from keras.models import Layer
from keras import backend
from grab_screen import get_scaled_grayscale, get_screen
from grab_key import get_keys
import numpy as np
import pyxinput
import os, sys
import time
import cv2

class ScaleLayer(Layer):
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
    def call(self, x):
        return (x-0.5)*2
    def compute_output_shape(self, input_shape):
        return input_shape
    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

def loss(y_true, y_pred): return backend.sum(backend.abs(y_true - y_pred))
def accuracy(y_true, y_pred): return 1 - backend.mean(backend.minimum(1.0,((backend.abs(y_true - y_pred))*10)))

file_number = int(sys.argv[1])
path_models = os.path.join(os.getcwd(), '../models')
model_name = 'model_' + '{0:03d}'.format(file_number)
model = load_model(path_models + '\\' + model_name + '.h5', custom_objects={'loss': loss, 'accuracy': accuracy, 'ScaleLayer': ScaleLayer})

print("Starting...\nPress O to pause\nPress L to stop")

wait_time = 0
for i in range(wait_time):
	print(wait_time-i)
	time.sleep(1)

start = time.time()
last_p_time = 0
counter = 0
paused = False

try:
    joystick = pyxinput.vController()
except MaxInputsReachedError:
    print('Unable to connect controller for testing.')

while joystick:
	keys = get_keys()
	if 'O' in keys:
		if time.time() - last_p_time > 0.5: 
			last_p_time = time.time()
			paused = not paused
			print('{:3.2f} Paused'.format(steering), end='\r')
			if paused == True: paused_time = time.time();
			if paused == False: start += time.time() - paused_time
	if 'L' in keys:
		print("\n\nExiting...")
		del joystick
		break
	if not paused:
		screen = get_screen()
		screen = cv2.resize(screen, (100, 75))
		screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
		# steering, throttle, brake = model.predict(screen[None, :])
		# joystick.set_value('AxisLx', steering)
		# joystick.set_value('TriggerR', throttle)
		# joystick.set_value('TriggerL', brake)
		# print('[{:3.2f}, {:3.2f}, {3.2f}]'.format(steering, throttle, brake), '@{0:4.2f}fps'.format(counter/(time.time() - start)) ,end='\r')

		steering = model.predict(screen[None, :,:, None])[0][0]
		joystick.set_value('AxisLx', steering)
		print('{:3.2f}'.format(steering), '@{0:4.2f}fps'.format(counter/(time.time() - start)), end='\r')

		counter += 1


