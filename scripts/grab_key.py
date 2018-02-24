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

global_keys = []

def get_keys():
	keys = []
	for key in keyList:
		if win32api.GetAsyncKeyState(ord(key)):
			keys.append(key)
	for keycode in keyCodeList:
		if win32api.GetAsyncKeyState(keycode):
			keys.append(keycode)
	global_keys.append(keys)
	return keys

def keys_to_label(keys):
	label = 0
	if 'W' in keys and 'A' in keys:		        label = 5 # WA
	elif 'W' in keys and 'D' in keys:	        label = 6 # WD
	elif 'S' in keys and 'A' in keys:	        label = 7 # SA
	elif 'S' in keys and 'D' in keys:	        label = 8 # SD
	elif 'W' in keys:	        				label = 1 # W
	elif 'S' in keys:	        				label = 2 # S
	elif 'A' in keys:	        				label = 3 # A
	elif 'D' in keys:	        				label = 4 # D
	return label

def get_global_keys():
	global global_keys

	labels = []
	for keys in global_keys:
		label = keys_to_label(keys)
		labels.append(label)

	return labels

if __name__ == '__main__':
	print(get_keys())
	