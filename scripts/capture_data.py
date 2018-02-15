from grab_screen import get_screen, save_image
from grab_key import get_keys, save_keys
import threading

print("Starting...\nPress P to stop")

import time
for i in range(5):
	print(5-i)
	time.sleep(1)

start = time.time()
counter = 0
while True:
	keys = get_keys()
	if 'P' in keys: break
	screen = get_screen()
	save_image(screen)
	counter += 1
	if counter%100 == 0:
		time_taken = time.time() - start
		print(counter/time_taken, "fps")

save_keys()
print("Files successfully saved :)")