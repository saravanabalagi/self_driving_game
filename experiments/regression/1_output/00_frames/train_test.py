from model import create_model
from save_load import save
from preprocess import get_test_train_data
import sys

# Print output to file
sys.stdout = open('log', 'w+')

def evaluate_model(model, x_test, y_test):
    print("\n\n")
    scores = model.evaluate(x_test, y_test)
    print("Accuracy: ", scores[1]*100, "%")
    return scores

if __name__ == '__main__':
	file = "F:\Projects\python\self_driving_game\data\dataset_mini.pz"
	x_train, x_test, y_train, y_test = get_test_train_data(file, 1000)

	model = create_model()
	model.fit(x_train, y_train, validation_split=0.1, epochs=25, batch_size=125)
	evaluate_model(model, x_test, y_test)
	save(model)