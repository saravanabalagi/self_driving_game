from send_input import press_key, release_key
from keras.models import load_model
from grab_screen import get_screen, save_image, get_scaled_grayscale
from grab_key import get_keys, save_keys
import numpy as np
import os, sys

# Keys
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

def send_key_input(key):
	repeat = 1
	if key == W or key == S:
		repeat = 10
	for i in range(repeat):
		press_key(key)
	# release_key(key)

def release_all_keys():
	release_key(W)
	release_key(S)
	release_key(A)
	release_key(D)

def label_to_key(label):
	if 	 label == 0: release_key(W); release_key(S); release_key(A); release_key(D)
	elif label == 5: release_all_keys(); send_key_input(W); send_key_input(A)
	elif label == 6: release_all_keys(); send_key_input(W); send_key_input(D)
	elif label == 7: release_all_keys(); send_key_input(S); send_key_input(A)
	elif label == 8: release_all_keys(); send_key_input(S); send_key_input(D)
	elif label == 1: release_all_keys(); send_key_input(W)
	elif label == 2: release_all_keys(); send_key_input(S)
	elif label == 3: release_all_keys(); send_key_input(A)
	elif label == 4: release_all_keys(); send_key_input(D)
	else: print("Label Error")


file_number = int(sys.argv[1])
path_models = os.path.join(os.getcwd(), '../models')
model_name = 'model_' + '{0:03d}'.format(file_number)
model = load_model(path_models + '\\' + model_name + '.h5')


print("Starting...\nPress P to stop")

import time
wait_time = 0
for i in range(wait_time):
	print(wait_time-i)
	time.sleep(1)

start = time.time()
counter = 0
while True:
	keys = get_keys()
	if 'P' in keys: break
	screen = get_scaled_grayscale(get_screen())
	label = np.argmax(model.predict(screen[None, None, :]))
	label_to_key(label)
	print(label)
	counter += 1
	if counter%100 == 0:
		time_taken = time.time() - start
		print(counter/time_taken, "fps")



