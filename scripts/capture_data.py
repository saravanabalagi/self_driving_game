from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario
from deepgtav.client import Client
from scipy.stats import mode
import argparse
import time
from tqdm import tqdm

weatherList = ["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "CLEARING", "THUNDER", "SMOG", "FOGGY", "XMAS", "SNOWLIGHT", "BLIZZARD", "NEUTRAL", "SNOW" ]
frame_capture_size = [800,600]
frame_save_size = [400,300]

def reset(weatherIndex=0):
	''' Resets position of car to a specific location '''
	# Same conditions as below | 
	client.sendMessage(Stop())
	dataset = Dataset(rate=30, frame=frame_capture_size, throttle=True, brake=True, steering=True, location=True, speed=True, yawRate=True, direction=True)
	# dataset = Dataset(rate=30, frame=[400,300], throttle=True, brake=True, steering=True, location=True, speed=True, yawRate=True, direction=True, reward=[18.0, 0.5])
	# Automatic driving scenario
	# scenario = Scenario(weather='EXTRASUNNY',vehicle='voltic',time=[12,0],drivingMode=[786603,70.0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]) 
	# scenario = Scenario(weather=weatherList[weatherIndex],vehicle='voltic',time=[12,0],drivingMode=[4294967295,70.0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]) 
	scenario = Scenario(weather=weatherList[weatherIndex], vehicle='voltic', time=[12,0], drivingMode=[2883621,20.0], wander=False) 
	client.sendMessage(Start(scenario=scenario,dataset=dataset)) # Start request

# Stores a pickled dataset file with data coming from DeepGTAV
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('-l', '--host', default='localhost', help='The IP 	where DeepGTAV is running')
	parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
	parser.add_argument('-d', '--dataset_path', default='dataset.pz', help='Place to store the dataset')
	args = parser.parse_args()

	# how many frames to skip in the beginning
	frames_to_skip = 500
	initial_pre_count = -(frames_to_skip-1)
	pre_count = initial_pre_count

	count = 226500
	no_of_images = 0
	weatherCount = 11
	to_remove = []
	speed = []
	direction = []
	pred_speed = None
	pred_dir = None

	# how much data to collect
	data_to_collect_per_weather = 20000
	rep_per_weather = 4
	reset_every = int(data_to_collect_per_weather / rep_per_weather)

	# Creates a new connection to DeepGTAV using the specified ip and port 
	client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9, frame_capture_size=frame_capture_size, frame_save_size=frame_save_size) 
	reset(weatherCount)

	print('Data to collect per weather:', data_to_collect_per_weather)
	print('Reset every:', reset_every, 'frames in each weather\n')
	print('{:>15} | {:^16} | {:^21} | {:^8} | {}'.format("Weather", "Progress", "[str, thtl, brk]", "Speed", "Direction"))

	while True: # Main loop
		try:
			# Message recieved as a Python dictionary
			message = client.recvMessage(pre_count+1>=0)
			if message is None: print('Message Error'); continue

			if pre_count<0: pre_count += 1
			if pre_count==0:
				count += 1
				no_of_images += 1

			new_location = message['location']
			
			print('{:>15}: {:>8d}/{:<8d} | [{: 3.2f}, {: 3.2f}, {: 3.2f}] | {:5.2f}/{:>2} | {} {:6.2f} | {:9}/{:< 4}'.format(weatherList[weatherCount], no_of_images, 
				data_to_collect_per_weather, message['steering'], message['throttle'], message['brake'], message['speed'], pred_speed or '?', message['direction'][0], message['direction'][1], count, pre_count), end='\r')
			
			if pre_count == initial_pre_count + 1:
				old_location = new_location

			# if (count % 100) == 0:
				# print(count, end='\r')

			# Checks if car is stuck, resets position if it is
			if (pre_count!=0 and pre_count%250==0) or (pre_count==-1) or (pre_count==0 and count%250==0):
				# Float position converted to ints so it doesn't have to be in the exact same position to be reset
				if int(new_location[0]) == int(old_location[0]) and int(new_location[1]) == int(old_location[1]) and int(new_location[2]) == int(old_location[2]):
					print('{:>15}: {:>8d}/{:<8d} | Car is stuck for frames {}. Resetting...                  '.format(weatherList[weatherCount], no_of_images, 
							data_to_collect_per_weather, [count-250, count]))
					to_remove.append([count-250, count])
					if pre_count==0: no_of_images -= 250
					if no_of_images < 0: 
						no_of_images = 0
					reset(weatherCount)
					pre_count = initial_pre_count
					pred_speed = None 
					pred_dir = None
					continue
				old_location = message['location']
				# print('At location: ' + str(old_location))

			# Check if the car is very slow
			if pre_count < -1: 
				speed.append(int(round(message['speed'])))
				direction.append(int(message['direction'][0]))
			if pre_count == -1:
				pred_speed = mode(speed).mode[0]
				pred_dir = mode(direction).mode[0]
				if pred_dir <= 1 or pred_speed < 8:
					print('{:>15}: {:>8d}/{:<8d} | Car is very slow {} or in wrong direction {}. Resetting...                  '.format(weatherList[weatherCount], no_of_images, 
								data_to_collect_per_weather, pred_speed, pred_dir))
					reset(weatherCount)
					pre_count = initial_pre_count
					pred_speed = None 
					pred_dir = None
				speed.clear()
				direction.clear()
				continue

			# reset once in a while
			if pre_count==0 and no_of_images > 0 and no_of_images % reset_every == 0:
				print("")
				reset(weatherCount)
				pre_count = initial_pre_count
				pred_speed = None 
				pred_dir = None
				continue

			# stopping criteria
			if no_of_images >= data_to_collect_per_weather: 
				no_of_images = 0
				weatherCount += 1
				if weatherCount > len(weatherList)-1: 
					break
				print("")
				reset(weatherCount)
				pre_count = initial_pre_count
				pred_speed = None 
				pred_dir = None
				continue

		except KeyboardInterrupt:
			i = input('\nPaused. Press p to continue and q to exit... ')
			if i == 'p':
				continue
			elif i == 'q':
				break

	# save to_remove
	with open('to_remove.txt', 'a+') as file:
		print(to_remove, file=file)
			
	# DeepGTAV stop message
	client.sendMessage(Stop())
	client.close() 
