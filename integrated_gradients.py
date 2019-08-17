from ltsm_baseline import *


def create_baseline_example(test_data):

	baseline =  np.zeros_like(test_data)+32 #F(x) is very close to 0
	assert(len(baseline[0] == len(test_data[0])))

	return baseline


def create_steps(test_data, baseline_data, step = 10):

	input_collection = []

	print("creating steps between X and X'")

	diff = baseline_data - test_data
	increment = diff/step
	input_collection.append(baseline_data)

	for i in range(step-2): #X..steps-2...X'
		temp = baseline_data+increment
		input_collection.append(temp)

	input_collection.append(test_data)

	assert(len(input_collection) == step)

	return input_collection


if __name__ == "__main__":

	# TODO remove the need for these annoying global definitions

	embed_size = 300
	batch_size = 1  # default was 250 for training

	num_layers = 1
	dropout = 0.5
	learning_rate = 0.001
	epochs = 1

	# stick to these parameters while training, restoring and predicting
	lstm_size = 64
	multiple_fc = False
	fc_units = 128

	checkpoint_to_restore = "/Users/sharan/Desktop/RNN_with_embd/64,False,128.ckpt"

	imdb_train_path = "./imdb/train.tsv"
	imdb_test_path = "./imdb/test.tsv"


	# for imdb dataset
	vocab_size = 99426


	tokenizer = load_tokenizer()

	test_data = create_test_example(tokenizer) # read from ./imdb/heatmap_test.tsv and import

	baseline_data = create_baseline_example(test_data)

	model, predictions = load_and_make_predictions_safe(lstm_size,
														multiple_fc,
														fc_units,
														vocab_size,
														embed_size,
														batch_size,
														num_layers,
														dropout,
														learning_rate,
														checkpoint_to_restore,
														baseline_data)

	inputs = create_steps(test_data, baseline_data)
	print(inputs)




