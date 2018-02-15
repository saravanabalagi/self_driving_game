import cv2
import os

path_src = os.path.join(os.getcwd(), '../data/source')
path_dest = os.path.join(os.getcwd(), '../data/processed')
if not os.path.exists(path_dest):
	os.makedirs(path_dest)

valid_images = [".jpg",".gif",".png",".tga"]
for file in os.listdir(path_src):
	ext = os.path.splitext(file)[1]
	if ext.lower() not in valid_images: continue
	img = cv2.imread(os.path.join(path_src,file))
	# imgs.append(Image.open(os.path.join(path,f))

	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = cv2.resize(img, (100, 75))
	cv2.imwrite(os.path.join(path_dest, file), img)