from PIL import ImageGrab
import numpy as np
import cv2
import os
import threading

# Constants
SCREEN_X_START = 0
SCREEN_Y_START = 0
SCREEN_X_END = 800
SCREEN_Y_END = 600

COUNTER = 1
PATH = os.path.join(os.getcwd(), '../data/source')


def get_screen(x_start=SCREEN_X_START, x_end=SCREEN_X_END, y_start=SCREEN_Y_START, y_end=SCREEN_Y_END):
	screen = np.array(ImageGrab.grab([x_start, y_start, x_end, y_end]))
	screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
	return screen

def save_image(img):
	global COUNTER
	threading.Thread(target=write_image, args=(img, COUNTER, )).start()
	COUNTER += 1

def write_image(img, number):
	global PATH
	if not os.path.exists(PATH):
		os.makedirs(PATH)
	screen_bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(os.path.join(PATH, 'screen_' + '{0:03d}'.format(number) + '.jpg'), img)

if __name__ == '__main__':
	save_image(get_screen())
	save_image(get_screen())