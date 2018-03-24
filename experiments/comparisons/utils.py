from keras import backend

def print_optimizer(model):
	print()
	print('{:>20} : {}'.format('Optimizer', type(model.optimizer).__name__))
	print('_________________________________________________________________\n')
	print('{:>20} : {:7.6f}'.format('lr', backend.get_value(model.optimizer.lr)))
	print('{:>20} : {:7.6f}'.format('beta_1', backend.get_value(model.optimizer.beta_1)))
	print('{:>20} : {:7.6f}'.format('beta_2', backend.get_value(model.optimizer.beta_2)))
	print('{:>20} : {}'.format('decay', backend.get_value(model.optimizer.decay)))
	print('{:>20} : {}'.format('amsgrad', model.optimizer.amsgrad))
	print('{:>20} : {}'.format('epsilon', model.optimizer.epsilon))
	print('{:>20} : {}'.format('initial_decay', model.optimizer.initial_decay))
	print('_________________________________________________________________\n')

def print_metrics(model):
	print('{:>20} : {}'.format('Loss Function', model.loss if not callable(model.loss) else model.loss.__name__))
	print('_________________________________________________________________\n')
	for i,metric in enumerate(model.metrics):
		print('{:>18} {} : {}'.format('Metric',i, metric if not callable(metric) else metric.__name__))
	print('_________________________________________________________________\n')

def print_model_summary(model):
	model.summary()
	print_optimizer(model)
	print_metrics(model)