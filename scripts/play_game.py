from send_input import label_to_keys
from keras.models import load_model
from grab_screen import get_screen, save_image, get_scaled_grayscale
from grab_key import get_keys, save_keys
import numpy as np
import os, sys
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
file_number = int(sys.argv[1])
path_models = os.path.join(os.getcwd(), '../models')
model_name = 'model_' + '{0:03d}'.format(file_number)
model = load_model(path_models + '\\' + model_name + '.h5')

print("Starting...\nPress O to pause\nPress L to stop")

import time
wait_time = 0
for i in range(wait_time):
	print(wait_time-i)
	time.sleep(1)

start = time.time()
last_p_time = 0
counter = 0
labels_q = deque()
paused = False

while True:
	keys = get_keys()
	if 'O' in keys:
		if time.time() - last_p_time > 0.5: 
			last_p_time = time.time()
			paused = not paused
			print(list(labels_q), 'Paused    ' end='\r')
			if paused == True: paused_time = time.time(); release_all_keys()
			if paused == False: start += time.time() - paused_time
	if 'L' in keys:
		print("Exiting...")
		break
	if not paused:
		screen = get_scaled_grayscale(get_screen())
		label = np.argmax(model.predict(screen[None, None, :]))
		label_to_keys(label)

		if len(list(labels_q)) > 15: labels_q.popleft()
		labels_q.append(label)
		print(list(labels_q), '@{0:4.2}fps'.format(counter/(time.time() - start)) ,end='\r')

		counter += 1


