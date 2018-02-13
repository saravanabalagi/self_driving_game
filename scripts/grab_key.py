import win32api, win32con
import time
import csv
import os

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'/\\":
	keyList.append(char)

arrow_up = win32con.VK_UP
arrow_down = win32con.VK_DOWN
arrow_left = win32con.VK_LEFT
arrow_right = win32con.VK_RIGHT

keyCodeList = []
keyCodeList.append(arrow_up)
keyCodeList.append(arrow_down)
keyCodeList.append(arrow_left)
keyCodeList.append(arrow_right)


def get_keys():
	keys = []
	for key in keyList:
		if win32api.GetAsyncKeyState(ord(key)):
			keys.append(key)
	for keycode in keyCodeList:
		if win32api.GetAsyncKeyState(keycode):
			keys.append(keycode)
	return keys

def save_keys(keys):
	path = os.path.join(os.getcwd(), '../data/')
	if not os.path.exists(path):
		os.makedirs(path)

	labels_file_path = os.path.join(path, 'labels.csv')
	with open(labels_file_path, 'w+', newline='') as labels_file:
		writer = csv.writer(labels_file)
		writer.writerow(keys)


if __name__ == '__main__':
	save_keys(get_keys())
	