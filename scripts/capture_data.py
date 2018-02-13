from grab_screen import get_screen, save_image
from grab_key import get_keys, save_keys

import time
time.sleep(5)

for i in range(10):
	keys = get_keys()
	screen = get_screen()

	save_image(screen)
	save_keys(keys)