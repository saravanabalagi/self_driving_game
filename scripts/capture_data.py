from grab_screen import get_screen, get_scaled_grayscale
from grab_key import get_keys, keys_to_label
from keras.models import load_model
from collections import Counter
from collections import deque
import numpy as np
import threading
import os, sys
import time
import cv2

from keras.utils import to_categorical
number_of_classes = 9

PATH_IMG = os.path.join(os.getcwd(), '../data/source')
PATH_KEY = os.path.join(os.getcwd(), '../data/labels.txt')
PATH_MODELS = os.path.join(os.getcwd(), '../models')

if not os.path.exists(PATH_IMG): os.makedirs(PATH_IMG)

def train(model, screens, labels):
	global number_of_classes
	model.fit(screens[:, None, ...], to_categorical(labels, number_of_classes), verbose=0)

def print_capture_stats(label, key_list, fps):
	global labels_q
	if len(list(labels_q)) > 10: labels_q.popleft()
	labels_q.append(label)
	print(list(labels_q), Counter(key_list), '@{0:4.2f}fps'.format(fps), end='\r')

def save_image(img, number, path):
	cv2.imwrite(os.path.join(path, 'screen_' + '{0:03d}'.format(number) + '.jpg'), img)

def save_key(label, file):
	print(label, file=file)

def get_last_file_number(modeL_name, path):
    numbers = [-1]
    for file in os.listdir(path):
        filename = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        if ext.lower() not in ['.h5']: continue
        if filename.startswith(model_name): 
            try: numbers.append(int(filename[len(model_name)+1:]))
            except: pass
    counter = max(numbers)
    return counter

def save_model(model, model_name, model_path):
	save_count = get_last_file_number(model_name, model_path)+1
	print("Saving", model_name+ '_{0:03d}'.format(save_count) +'.h5', end='\r')
	model.save(model_path + '\\' + model_name + '_{0:03d}'.format(save_count) +'.h5')
	print(model_name+ '_{0:03d}'.format(save_count) +'.h5', "saved successfully")
	save_count += 1


# Load model if necessary
if(len(sys.argv)==1): 
	print("No training will take place.\nMode: Capture, Save\n\n")
	training_enabled = False
else:
	model_number = int(sys.argv[1])
	if(len(sys.argv)==3):
		model_number_ext = int(sys.argv[2])
		model_name = 'model_' + '{0:03d}'.format(model_number) + '{0:03d}'.format(model_number_ext)
	else: model_name = 'model_' + '{0:03d}'.format(model_number)
	model_name_without_ext = 'model_' + '{0:03d}'.format(model_number)
	print("Loading", model_name, end='\r')
	model = load_model(PATH_MODELS + '\\' + model_name + '.h5')
	print(model_name, "loaded successfully")
	print("Mode: Capture, Train, Save\n\nCapture Stats:")
	training_enabled = True

# Wait for 5 seconds before capture
wait_time = 5
for i in range(wait_time):
	print("Starting in",wait_time-i, end='\r')
	time.sleep(1)
print("Press O to pause\nPress L to stop\n")

# Main capture loop
start = time.time()
last_p_time = 0
paused = False
counter = 0
labels_q = deque()
labels_file = open(PATH_KEY, 'w+')
keys_list = []

batch_size = 125
keys_training_batch = []
screens_training_batch = []

while True:
	keys = get_keys()
	if 'O' in keys:
		if time.time() - last_p_time > 0.5: 
			last_p_time = time.time()
			paused = not paused
			print(list(labels_q), Counter(keys_list), 'Paused   ', end='\r')
			if paused == True: paused_time = time.time();
			if paused == False: start += time.time() - paused_time
	if 'L' in keys:
		print("\n\nExiting...")
		break

	if not paused:

		#grab screen and key
		screen = get_screen()
		label = keys_to_label(keys)
		keys_list.append(label)

		# print capture stats
		fps = counter/(time.time()-start)
		print_capture_stats(label, keys_list, fps)

		# save image and key
		threading.Thread(target=save_image, args=(screen, counter, PATH_IMG,)).start()
		threading.Thread(target=save_key, args=(keys_to_label(keys), labels_file,)).start()

		# train model
		if training_enabled:
			screen_resized = get_scaled_grayscale(screen)
			screens_training_batch.append(screen_resized)
			keys_training_batch.append(label)
			if (counter+1)%batch_size==0:
				threading.Thread(target=train, args=(model, np.array(screens_training_batch), np.array(keys_training_batch),)).start()
				keys_training_batch = []
				screens_training_batch = []

		counter += 1

labels_file.close()
if training_enabled:
	save_model(model, model_name_without_ext, PATH_MODELS)
