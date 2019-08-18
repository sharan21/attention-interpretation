from ltsm_baseline import *


def create_baseline_example(test_data):# empirical for NLP

	print("Creating Baseline example...'")

	baseline =  np.zeros_like(test_data)+35 #F(x) is very close to 0
	assert(len(baseline[0] == len(test_data[0])))

	return baseline


def create_input_collection(test_data, baseline_data, step = 11): # will return step + 1 inputs

	input_collection = []

	print("Creating steps between X and X...'")

	diff = test_data[0] - baseline_data[0]
	temp = baseline_data[0]

	increment = diff/step
	# print("inc is {}".format(diff))
	input_collection.append(baseline_data[0])

	for i in range(step-1): #X..steps-2...X'
		temp = [int(e) for e in temp+increment]
		input_collection.append(np.array(temp))

	input_collection.append(test_data[0])

	assert(len(input_collection) == step+1)

	return input_collection


# TODO Refactor below function to use list arg or *args
def run_one_grad(lstm_size, multiple_fc, fc_units, vocab_size, embed_size, batch_size, num_layers, dropout, learning_rate, checkpoint_to_restore, test_data):
	#finds gradients for one instance of test_data

	print("Computing grads for input...'")

	model, all_preds = load_and_make_predictions_withargs(lstm_size, multiple_fc, fc_units,
															  vocab_size, embed_size, batch_size,
															  num_layers, dropout, learning_rate,
															  checkpoint_to_restore, test_data)

	grads = get_gradients(model, all_preds, test_data)
	# print("computed gradients tensor: {}".format(grads))

	grads_list, indices_list = get_gradients_values(grads)

	text_list = get_word_from_index(indices_list, tokenizer)
	word_list = [e for e in text_list[0].split(" ")]

	# print("indices list: {}".format(indices_list))
	# print("words from indices: {}".format(text_list))

	compressed_grads = compress_gradients(grads_list)

	grads_cleaned = clean_attributes(compressed_grads, word_list)

	# print(attributions_dict)

	return grads_list, compressed_grads , grads_cleaned




def compute_integrated_gradients(input_collection):

	print("Computing IG for inputs collection")
	global lstm_size, multiple_fc, fc_units, vocab_size, embed_size, batch_size, num_layers, dropout, learning_rate, checkpoint_to_restore

	x_dash = input_collection[0]
	x = input_collection[-1]
	print(x_dash)

	diff = x - x_dash
	integral = np.zeros_like(x)

	print("x - x_dash : {}".format(diff))
	print("Integral init: {}".format(sum))


	for i in range(len(input_collection)):
		input = input_collection[i]
		buffer = []
		print("input is: {}".format(input))

		buffer.append(input)


		grads, attri, attri_dict = run_one_grad(lstm_size, multiple_fc, fc_units,
												vocab_size, embed_size, batch_size,
												num_layers, dropout, learning_rate,
												checkpoint_to_restore, buffer)


		integral = np.add(integral, attri)
		intergrated_grads = np.multiply(diff, integral)

	return intergrated_grads

















if __name__ == "__main__":

	# TODO remove the need for these annoying global definitions

	embed_size = 300
	batch_size = 1  # default was 250 for training
	steps = 11

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
	print("test data: {}".format(test_data))
	test_data_sent = get_word_from_index(test_data[0], tokenizer)
	print("test sentence: {}".format(test_data_sent))
	test_data_words = [e for e in test_data_sent[0].split(" ")]
	print("test sentence word list: {}".format(test_data_words))
	baseline_data = create_baseline_example(test_data)
	print("baseline data: {}".format(baseline_data))


	input_collection = create_input_collection(test_data, baseline_data)  #returns a list of integer lists
	# we now have all input vectors for IG

	integ_grads = compute_integrated_gradients(input_collection)

	integ_grads_cleaned = clean_attributes(integ_grads, test_data_words)



	grads_list, compressed_grads, grads_cleaned = run_one_grad(lstm_size, multiple_fc, fc_units,
						 vocab_size, embed_size, batch_size,
					 num_layers, dropout, learning_rate,
					 checkpoint_to_restore, test_data)

	print(integ_grads_cleaned)
	print(grads_cleaned)
















