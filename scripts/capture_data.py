from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario
from deepgtav.client import Client
import argparse
import time
from tqdm import tqdm

weatherList = ["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "CLEARING", "THUNDER", "SMOG", "FOGGY", "XMAS", "SNOWLIGHT", "BLIZZARD", "NEUTRAL", "SNOW" ]

def reset(weatherIndex=0):
	''' Resets position of car to a specific location '''
	# Same conditions as below | 
	client.sendMessage(Stop())
	dataset = Dataset(rate=30, frame=[400,300], throttle=True, brake=True, steering=True, location=True, speed=True, yawRate=True)
	# Automatic driving scenario
	# scenario = Scenario(weather='EXTRASUNNY',vehicle='voltic',time=[12,0],drivingMode=[786603,70.0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]) 
	# scenario = Scenario(weather=weatherList[weatherIndex],vehicle='voltic',time=[12,0],drivingMode=[4294967295,70.0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]) 
	scenario = Scenario(weather=weatherList[weatherIndex], vehicle='voltic', time=[12,0], drivingMode=[786603,20.0]) 
	client.sendMessage(Start(scenario=scenario,dataset=dataset)) # Start request

# Stores a pickled dataset file with data coming from DeepGTAV
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
	parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
	parser.add_argument('-d', '--dataset_path', default='dataset.pz', help='Place to store the dataset')
	args = parser.parse_args()

	# Creates a new connection to DeepGTAV using the specified ip and port 
	client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9) 
	reset()
	
	count = 0
	no_of_images = 0
	weatherCount = 0
	to_remove = []
	data_to_collect_per_weather = 20000
	reset_every = int(data_to_collect_per_weather / 8)
	print('{:>15} | {:^16} | {:^21} | {}'.format("Weather", "Progress", "[str, thtl, brk]", "Speed"))

	while True: # Main loop
		try:
			# Message recieved as a Python dictionary
			message = client.recvMessage()
			new_location = message['location']

			count += 1
			no_of_images += 1
			print('{:>15}: {:>8d}/{:<8d} | [{: 3.2f}, {: 3.2f}, {: 3.2f}] | {:5.2f}'.format(weatherList[weatherCount], no_of_images, 
				data_to_collect_per_weather, message['steering'], message['throttle'], message['brake'], message['speed']), end='\r')
			
			if no_of_images == 1:
				old_location = new_location

			# if (count % 100) == 0:
				# print(count, end='\r')

			# Checks if car is stuck, resets position if it is
			if (count % 250)==0:
				# Float position converted to ints so it doesn't have to be in the exact same position to be reset
				if int(new_location[0]) == int(old_location[0]) and int(new_location[1]) == int(old_location[1]) and int(new_location[2]) == int(old_location[2]):
					print('{0:>15}: {1:>8d}/{2:<8d} | Car is stuck for frames {3}. Resetting...                  '.format(weatherList[weatherCount], no_of_images, 
							data_to_collect_per_weather, [count-250, count]))
					to_remove.append([count-250, count])
					no_of_images -= 250
					reset(weatherCount)
				old_location = message['location']
				# print('At location: ' + str(old_location))

			# reset once in a while
			if no_of_images % reset_every == 0:
				reset(weatherCount)

			# stopping criteria
			if no_of_images >= data_to_collect_per_weather: 
				no_of_images = 0
				weatherCount += 1
				if weatherCount > len(weatherList)-1: 
					break
				print("")
				reset(weatherCount)


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
