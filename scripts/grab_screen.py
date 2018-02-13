from PIL import ImageGrab
import numpy as np
import cv2
import os

# Constants
SCREEN_X_START = 0
SCREEN_Y_START = 0
SCREEN_X_END = 800
SCREEN_Y_END = 600

COUNTER = 1

def get_screen(x_start=SCREEN_X_START, x_end=SCREEN_X_END, y_start=SCREEN_Y_START, y_end=SCREEN_Y_END):
	screen = np.array(ImageGrab.grab([x_start, y_start, x_end, y_end]))
	screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
	return screen

def save_image(img):
	global COUNTER
	path = os.path.join(os.getcwd(), '../data/')
	if not os.path.exists(path):
	    os.makedirs(path)

	screen = img
	screen_bg = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(os.path.join(path, 'screen_' + '{0:03d}'.format(COUNTER) + '.jpg'), screen)
	COUNTER += 1
	# cv2.imwrite(os.path.join(path, 'screen_bg_' + '{0:03d}'.format(COUNTER) +'.jpg'), screen_bg)

if __name__ == '__main__':
	save_image(get_screen())
	save_image(get_screen())