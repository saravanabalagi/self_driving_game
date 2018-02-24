from grab_screen import get_screen, get_scaled_grayscale
from grab_key import get_keys, get_global_keys, keys_to_label
from keras.models import load_model
from collections import Counter
from collections import deque
import numpy as np
import threading
import os, sys
import time
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras.utils import to_categorical
number_of_classes = 9

PATH_IMG = os.path.join(os.getcwd(), '../data/source')
PATH_KEY = os.path.join(os.getcwd(), '../data/labels.txt')
if not os.path.exists(PATH_IMG): os.makedirs(PATH_IMG)

def train(model, screen, label):
	global number_of_classes
	model.fit(screen[None, None, :], to_categorical(label, number_of_classes)[None, :], verbose=0)

def print_capture_stats(label, key_list, fps):
	global labels_q
	if len(list(labels_q)) > 10: labels_q.popleft()
	labels_q.append(label)
	print(list(labels_q), Counter(key_list), '@{0:4.2f}fps'.format(fps), end='\r')

def save_image(img, number, path):
	cv2.imwrite(os.path.join(path, 'screen_' + '{0:03d}'.format(number) + '.jpg'), img)

def save_key(label, file):
	print(label, file=file)


# Wait for 5 seconds before capture
wait_time = 5
for i in range(wait_time):
	print("Starting in",wait_time-i, end='\r')
	time.sleep(1)
print("Press O to pause\nPress L to stop\n")

# Load model if necessary
if(len(sys.argv)==1): 
	print("No training will take place.\nMode: Capture, Save\n\n")
	training_enabled = False
else:
	file_number = int(sys.argv[1])
	path_models = os.path.join(os.getcwd(), '../models')
	model_name = 'model_' + '{0:03d}'.format(file_number)
	print("Loading", model_name, end='\r')
	model = load_model(path_models + '\\' + model_name + '.h5')
	print(model_name, "loaded successfully")
	print("Mode: Capture, Train, Save\n\nCapture Stats:")
	training_enabled = True

# Main capture loop
start = time.time()
last_p_time = 0
paused = False
counter = 0
labels_q = deque()
labels_file = open(PATH_KEY, 'w+')

while True:
	keys = get_keys()
	if 'O' in keys:
		if time.time() - last_p_time > 0.5: 
			last_p_time = time.time()
			paused = not paused
			print(list(labels_q), Counter(get_global_keys()), 'Paused   ', end='\r')
			if paused == True: paused_time = time.time();
			if paused == False: start += time.time() - paused_time
	if 'L' in keys:
		print("\n\nExiting...")
		break

	if not paused:

		#grab screen and key
		screen = get_screen()
		label = keys_to_label(keys)

		# print capture stats
		fps = counter/(time.time()-start)
		print_capture_stats(label, get_global_keys(), fps)

		# save image and key
		threading.Thread(target=save_image, args=(screen, counter, PATH_IMG,)).start()
		threading.Thread(target=save_key, args=(keys_to_label(keys), labels_file,)).start()

		# train model
		if training_enabled:
			screen_resized = get_scaled_grayscale(screen)
			threading.Thread(target=train, args=(model, screen_resized, label,)).start()

		counter += 1

labels_file.close()
print("Files successfully saved :)")

