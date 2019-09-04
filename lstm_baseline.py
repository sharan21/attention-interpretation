import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import re
import math
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple
from rnn_cell import *



tf.logging.set_verbosity(tf.logging.ERROR)


def clean_text(text, remove_stopwords=True):
	'''Clean the text, with the option to remove stopwords'''

	# Convert words to lower case and split them
	text = text.lower().split()

	# Optionally, remove stop words
	if remove_stopwords:
		stops = set(stopwords.words("english"))
		text = [w for w in text if not w in stops]

	text = " ".join(text)

	# Clean the text
	text = re.sub(r"<br />", " ", text)
	text = re.sub(r"[^a-z]", " ", text)
	text = re.sub(r"   ", " ", text)  # Remove any extra spaces
	text = re.sub(r"  ", " ", text)

	# Remove punctuation from text
	text = ''.join([c for c in text if c not in punctuation])
	# Return a list of words
	return (text)

def get_batches(x, y, batch_size):
	'''Create the batches for the training and validation data'''
	n_batches = len(x) // batch_size
	x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
	for ii in range(0, len(x), batch_size):
		yield x[ii:ii + batch_size], y[ii:ii + batch_size]

def get_test_batches(x, batch_size):
	'''Create the batches for the testing data'''
	n_batches = len(x) // batch_size
	x = x[:n_batches * batch_size]
	for ii in range(0, len(x), batch_size):
		yield x[ii:ii + batch_size]

def get_gradients(model, predicted_y, test_data,  dropout = 0.5):


	# VIP!!!
	# ensure batch_size of loaded model == number of test cases


	optimizer_here = model.gradients
	embedding_here = model.embedding
	cost_here = model.cost

	gradients, variables = zip(*optimizer_here.compute_gradients(cost_here, embedding_here, gate_gradients=True, colocate_gradients_with_ops=True))
	# print("gradients object: {}".format(gradients[0]))

	with tf.Session() as sess:

		init = tf.global_variables_initializer()

		sess.run(init)
		test_state = sess.run(model.initial_state)


		# feed = {model.inputs: x_test[0:len(predicted_y)], # dims should match predicted_y
		# 		model.labels: predicted_y[:, None], #converting 1d to 2d array
		# 		model.keep_prob: dropout,
		# 		model.initial_state: test_state}

		feed = {model.inputs: test_data,  # dims should match predicted_y
				model.labels: predicted_y[:, None],  # converting 1d to 2d array
				model.keep_prob: dropout,
				model.initial_state: test_state}


		gradients_fed = sess.run(gradients, feed_dict=feed)


	return gradients_fed

def get_gradients_values(gradients): # takes IndexedSlices Object which store gradients as input

	index_slices = gradients[0]

	vals = index_slices[0]
	indices = index_slices[1]

	#saving the gradients object
	# np.savetxt("./vals.csv", vals, delimiter=",")
	# np.savetxt("./indices.csv", indices, delimiter=",")

	return vals, indices # val and indices are numpy arrays





def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers,
			  dropout, learning_rate, multiple_fc, fc_units, with_embd = True):
	'''Build the Recurrent Neural Network'''


	# Declare placeholders we'll feed into the graph
	with tf.name_scope('inputs'):
		inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

	with tf.name_scope('labels'):
		labels = tf.placeholder(tf.int32, [None, None], name='labels')

	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	# Create the embeddings
	with tf.name_scope("embeddings"): # "embeddings" for old models
		embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
		embed = tf.nn.embedding_lookup(embedding, inputs)

	# Build the RNN layers
	with tf.name_scope("RNN_layers"):
		lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
		cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

	# Set the initial state
	with tf.name_scope("RNN_init_state"):
		initial_state = cell.zero_state(batch_size, tf.float32)

	# Run the data through the RNN layers
	with tf.name_scope("RNN_forward"):
		outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

	# Create the fully connected layers
	with tf.name_scope("fully_connected"):
		# Initialize the weights and biases
		weights = tf.truncated_normal_initializer(stddev=0.1)
		biases = tf.zeros_initializer()

		dense = tf.contrib.layers.fully_connected(outputs[:, -1],num_outputs=fc_units,activation_fn=tf.sigmoid,weights_initializer=weights,biases_initializer=biases)
		dense = tf.contrib.layers.dropout(dense, keep_prob)

		# Depending on the iteration, use a second fully connected layer
		if multiple_fc == True:
			dense = tf.contrib.layers.fully_connected(dense,num_outputs=fc_units,activation_fn=tf.sigmoid,weights_initializer=weights,biases_initializer=biases)
			dense = tf.contrib.layers.dropout(dense, keep_prob)

	# Make the predictions
	with tf.name_scope('predictions'):
		predictions = tf.contrib.layers.fully_connected(dense,num_outputs=1,activation_fn=tf.sigmoid,weights_initializer=weights,biases_initializer=biases)
		tf.summary.histogram('predictions', predictions)

	# Calculate the cost
	with tf.name_scope('cost'):
		cost = tf.losses.mean_squared_error(labels, predictions)
		tf.summary.scalar('cost', cost)

	# Train the model
	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
		opt = tf.train.AdamOptimizer(learning_rate)

	with tf.name_scope('gradients'):
		gradients = tf.train.AdamOptimizer(learning_rate)

	# Determine the accuracy
	with tf.name_scope("accuracy"):
		correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

	# Merge all of the summaries
	merged = tf.summary.merge_all()

	# Export the nodes

	if(with_embd):

		export_nodes = ['inputs', 'labels','embedding', 'embed', 'keep_prob', 'initial_state', 'final_state', 'accuracy',
						'predictions', 'cost', 'optimizer', 'gradients','merged']
	else:
		export_nodes = ['inputs', 'labels', 'keep_prob', 'initial_state', 'final_state', 'accuracy',
						'predictions', 'cost', 'optimizer', 'gradients', 'merged']


	Graph = namedtuple('Graph', export_nodes)
	local_dict = locals()
	# print(local_dict)
	graph = Graph(*[local_dict[each] for each in export_nodes])

	return graph

def train_model(model, epochs, log_string, checkpoint_to_create):
	'''Train the RNN'''

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Used to determine when to stop the training early
		valid_loss_summary = []

		# Keep track of which batch iteration is being trained
		iteration = 0

		print()
		print("Training Model: {}".format(log_string))

		train_writer = tf.summary.FileWriter('./logs/3/train/{}'.format(log_string), sess.graph)
		valid_writer = tf.summary.FileWriter('./logs/3/valid/{}'.format(log_string))

		for e in range(epochs):
			state = sess.run(model.initial_state)

			# Record progress with each epoch
			train_loss = []
			train_acc = []
			val_acc = []
			val_loss = []

			with tqdm(total=len(x_train)) as pbar:
				for _, (x, y) in enumerate(get_batches(x_train, y_train, batch_size), 1):
					feed = {model.inputs: x,
							model.labels: y[:, None],
							model.keep_prob: dropout,
							model.initial_state: state}
					summary, loss, acc, state, _ = sess.run([model.merged,model.cost,model.accuracy,model.final_state,model.optimizer],feed_dict=feed)

					# Record the loss and accuracy of each training batch
					train_loss.append(loss)
					train_acc.append(acc)

					# Record the progress of training
					train_writer.add_summary(summary, iteration)

					iteration += 1
					pbar.update(batch_size)

			# Average the training loss and accuracy of each epoch
			avg_train_loss = np.mean(train_loss)
			avg_train_acc = np.mean(train_acc)

			val_state = sess.run(model.initial_state)
			with tqdm(total=len(x_valid)) as pbar:
				for x, y in get_batches(x_valid, y_valid, batch_size):
					feed = {model.inputs: x,
							model.labels: y[:, None],
							model.keep_prob: 1,
							model.initial_state: val_state}
					summary, batch_loss, batch_acc, val_state = sess.run([model.merged, model.cost, model.accuracy,
																		  model.final_state],
																		 feed_dict=feed)

					# Record the validation loss and accuracy of each epoch
					val_loss.append(batch_loss)
					val_acc.append(batch_acc)
					pbar.update(batch_size)

			# Average the validation loss and accuracy of each epoch
			avg_valid_loss = np.mean(val_loss)
			avg_valid_acc = np.mean(val_acc)
			valid_loss_summary.append(avg_valid_loss)

			# Record the validation data's progress
			valid_writer.add_summary(summary, iteration)

			# Print the progress of each epoch
			print("Epoch: {}/{}".format(e, epochs),
				  "Train Loss: {:.3f}".format(avg_train_loss),
				  "Train Acc: {:.3f}".format(avg_train_acc),
				  "Valid Loss: {:.3f}".format(avg_valid_loss),
				  "Valid Acc: {:.3f}".format(avg_valid_acc))

			# Stop training if the validation loss does not decrease after 3 epochs
			if avg_valid_loss > min(valid_loss_summary):
				print("No Improvement.")
				stop_early += 1
				if stop_early == 3: # set to 1 to prematuraly break
					break

				# Reset stop_early if the validation loss finds a new low
			# Save a checkpoint of the model
			else:
				print("New Record!")
				stop_early = 0


				# checkpoint = "/Users/sharan/Desktop/RNN_with_embd/{}.cptk".format(
				# 	log_string)
				saver.save(sess, checkpoint_to_create)

def load_and_make_predictions_batch(lstm_size, multiple_fc, fc_units, vocab_size, checkpoint, x_test):
	'''Predict the sentiment of the testing data'''

	pruning_size = 250 # ensure that pruning_size == batch_size while getting gradients
	x_test_pruned = x_test[0:pruning_size]
	all_preds = []

	model = build_rnn(n_words=vocab_size,
					  embed_size=embed_size,
					  batch_size=batch_size,
					  lstm_size=lstm_size,
					  num_layers=num_layers,
					  dropout=dropout,
					  learning_rate=learning_rate,
					  multiple_fc=multiple_fc,
					  fc_units=fc_units) # default with_embd = True

	with tf.Session() as sess:
		saver = tf.train.Saver()
		# Load the model
		saver.restore(sess, checkpoint)
		test_state = sess.run(model.initial_state)
		for _, x in enumerate(get_test_batches(x_test_pruned, batch_size=250), 1):
			feed = {model.inputs: x,
					model.keep_prob: 1,
					model.initial_state: test_state}
			predictions = sess.run(model.predictions, feed_dict=feed)
			for p in predictions:
				all_preds.append(float(p))
				print("prediction is :{}".format(p))


	return model, np.array(all_preds)

def load_and_make_predictions_single(lstm_size, multiple_fc, fc_units, vocab_size, checkpoint, test_data):
	'''Predict the sentiment of the testing data'''


	print("Size of test data: {}".format(len(test_data)))

	all_preds = []

	model = build_rnn(n_words=vocab_size,
					  embed_size=embed_size,
					  batch_size=batch_size,
					  lstm_size=lstm_size,
					  num_layers=num_layers,
					  dropout=dropout,
					  learning_rate=learning_rate,
					  multiple_fc=multiple_fc,
					  fc_units=fc_units) # default with_embd = True

	with tf.Session() as sess:
		saver = tf.train.Saver()
		# Load the model
		saver.restore(sess, checkpoint)
		test_state = sess.run(model.initial_state)
		for _, x in enumerate(get_test_batches(test_data, batch_size=batch_size), 1):
			feed = {model.inputs: x,
					model.keep_prob: 1,
					model.initial_state: test_state}
			predictions = sess.run(model.predictions, feed_dict=feed)
			for p in predictions:
				all_preds.append(float(p))
				print("Prediction :{}".format(p))


	return model, np.array(all_preds)

def load_and_make_predictions_withargs(lstm_size, multiple_fc, fc_units, vocab_size, embed_size, batch_size, num_layers, dropout, learning_rate, checkpoint, test_data):
	'''Predict the sentiment of the testing data'''


	print("Size of test data: {}".format(len(test_data)))

	all_preds = []

	tf.reset_default_graph() # fixes the Model already in use problem.

	model = build_rnn(n_words=vocab_size,
					  embed_size=embed_size,
					  batch_size=batch_size,
					  lstm_size=lstm_size,
					  num_layers=num_layers,
					  dropout=dropout,
					  learning_rate=learning_rate,
					  multiple_fc=multiple_fc,
					  fc_units=fc_units) # default with_embd = True

	with tf.Session() as sess:
		saver = tf.train.Saver()
		# Load the model
		saver.restore(sess, checkpoint)
		test_state = sess.run(model.initial_state)
		for _, x in enumerate(get_test_batches(test_data, batch_size=batch_size), 1):
			feed = {model.inputs: x,
					model.keep_prob: 1,
					model.initial_state: test_state}
			predictions = sess.run(model.predictions, feed_dict=feed)
			for p in predictions:
				all_preds.append(float(p))
				print("Prediction :{}".format(p))


	return model, np.array(all_preds)

def write_submission(predictions, string):

	global train, test

	submission = pd.DataFrame(data={"id": test["id"], "sentiment": predictions})
	submission.to_csv("submission_{}.csv".format(string), index=False, quoting=3)

def load_model(path, no_of_unique_words):

	tf.reset_default_graph()

	model = build_rnn(n_words=no_of_unique_words, embed_size=embed_size, batch_size=batch_size, lstm_size=lstm_size,
					  num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, multiple_fc=multiple_fc,
					  fc_units=fc_units)

	# saver = tf.train.import_meta_graph(path, clear_devices=True)
	saver = tf.train.Saver()

	with tf.Session() as sess:

		print("Restoring model from path {}".format(path))
		saver.restore(sess, path)

		print("Model restored and ready to use.")

	return model

def save_tokenizer(tokenizer):

	print("saving tokenizer...")
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(path = './tokenizers/tokenizer_imdb.pickle'):

	print("loading tokenizer...")
	with open(path, 'rb') as handle:
		tokenizer = pickle.load(handle)

	return tokenizer

def import_clean_tokenize_data(trainpath, testpath):

	global train, test, number_of_unique_words
	train = pd.read_csv(trainpath, delimiter="\t")
	test = pd.read_csv(testpath, delimiter="\t")

	train_clean = []
	for review in train.review:
		train_clean.append(clean_text(review))

	test_clean = []
	for review in test.review:
		test_clean.append(clean_text(review))

	all_reviews = train_clean + test_clean
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(all_reviews)
	print("Fitting of local vocabulary into tokenizer is complete.")

	train_seq = tokenizer.texts_to_sequences(train_clean)
	print("Train data tokenization is complete.")

	test_seq = tokenizer.texts_to_sequences(test_clean)
	print("Test data tokenization is complete")

	word_index = tokenizer.word_index
	print("Number of unique words in index: %d" % len(word_index))

	no_of_unique_words = len(word_index) + 1 # we need this explicit parameter to build the model

	return  train_seq, test_seq, no_of_unique_words, tokenizer

def pad_split_data(train_tokenized, test_tokenized):

	global train, test

	max_review_length = 200

	train_pad = pad_sequences(train_tokenized, maxlen=max_review_length)
	print("train_pad is complete.")

	test_pad = pad_sequences(test_tokenized, maxlen=max_review_length)
	print("test_pad is complete.")

	x_train, x_valid, y_train, y_valid = train_test_split(train_pad, train.sentiment, test_size=0.15, random_state=2)
	x_test = test_pad

	return  x_train, x_valid, y_train, y_valid, x_test

def train_and_checkpoint(checkpoint_to_create, l,m,f, vocab_size):


	log_string = 'ru={},fcl={},fcu={}'.format(l,m,f)

	model = build_rnn(n_words=vocab_size, embed_size=embed_size, batch_size=batch_size, lstm_size=l, num_layers=num_layers,
					  dropout=dropout, learning_rate=learning_rate, multiple_fc=m, fc_units=f, with_embd=True)

	train_model(model, epochs, log_string, checkpoint_to_create)

def checkpoint_to_vars(checkpoint):# DO NOT USE
	# returns ltsm_size, multiple_fc, and fc_units,

	filename = checkpoint.split("/")
	filename = filename[len(filename)-1]
	var_list = (filename.split("."))[0].split(",")

	return var_list[0], var_list[1], var_list[2]

def sort_dict(dict):
	buffer = [e for e in dict.items()]
	buffer.sort()

	return {x: y for x,y in buffer}

def create_test_example(tokenizer): # to call this we first ought to have fed entire dataset into the tokenizer earlier

	df = pd.read_csv("./datasets/imdb/single_test_data.tsv", delimiter="\t")

	print("Test example :\n {}".format(df['review']))

	test_example_clean = []
	for review in df.review:
		test_example_clean.append(clean_text(review))


	test_example_seq= tokenizer.texts_to_sequences(test_example_clean)
	print("Test data tokenization is complete")



	return test_example_seq, df

def compress_gradients(grads):

	return np.sum(grads, axis=1)

def get_start_point(indices):
	for i in range(len(indices)):
		if(indices[i] != 0):
			return(i)

def get_word_from_index(indices, tokenizer):
	text = tokenizer.sequences_to_texts([indices])
	return text

def clean_attributes(attributions, word_list):


	tot_sum = sum(abs(attributions))
	percent = [(abs(e)/tot_sum)*100 for e in attributions]

	attri_dict = dict(zip(word_list, percent))

	# assert(math.ceil(sum(percent) or  == 100)

	return sort_dict(attri_dict)

def predict_single_from_model(model, checkpoint, test_data):


	with tf.Session() as sess:
		saver = tf.train.Saver()
		# Load the model
		saver.restore(sess, checkpoint)
		test_state = sess.run(model.initial_state)
		feed = {model.inputs: test_data,
				model.keep_prob: 1,
				model.initial_state: test_state}
		prediction = sess.run(model.predictions, feed_dict=feed)

		return prediction



if __name__ == '__main__':

########################################################################## EXPERIMENT SECTION


	# embed_size = 300
	# batch_size = 1  # default was 250 for training
	#
	# num_layers = 1
	# dropout = 0.5
	# learning_rate = 0.001
	# epochs = 1
	#
	# # stick to these parameters while training, restoring and predicting
	# lstm_size = 64
	# multiple_fc = False
	# fc_units = 128
	#
	# checkpoint_to_create = "/Users/sharan/Desktop/RNN_with_embed/64,False,128.ckpt"
	# checkpoint_to_restore = "./models/LSTM_models/64,False,128/64,False,128.ckpt"
	#
	# imdb_train_path = "./datasets/imdb/train.tsv"
	# imdb_test_path = "./datasets/imdb/test.tsv"
	#
	# vocab_size = 99426
	#
	# tokenizer = load_tokenizer('./tokenizers/tokenizer_imdb.pickle')
	#
	# test_data = create_test_example(tokenizer)  # creates embeddings for test sentences using tokenizer from import...()
	#
	# model = build_rnn(n_words = vocab_size, embed_size = embed_size, batch_size = batch_size, lstm_size = lstm_size, num_layers = num_layers,
	# 		  dropout = dropout, learning_rate = learning_rate, multiple_fc = multiple_fc, fc_units = fc_units)
	#
	# # model, all_preds = load_and_make_predictions_single(lstm_size, multiple_fc, fc_units, vocab_size, checkpoint_to_restore,
	# # 													test_data)
	#
	# single_prediction = predict_single_from_model(model=model, checkpoint=checkpoint_to_restore, test_data=test_data)
	# print(single_prediction)



########################################################################## IMPORT DATA AND FIT TOKENIZER


	# embed_size = 300
	# batch_size = 250 #default was 250 for training
	#
	# num_layers = 1
	# dropout = 0.5
	# learning_rate = 0.001
	# epochs = 1
	#
	# #stick to these parameters while training, restoring and predicting
	# lstm_size = 64
	# multiple_fc = False
	# fc_units = 128
	#
	# checkpoint_to_create= "/Users/sharan/Desktop/RNN_with_embed/64,False,128.ckpt"
	# checkpoint_to_restore = "./models/LSTM_models/64,False,128/64,False,128.ckpt"
	#
	#
	# imdb_train_path = "./datasets/imdb/train.tsv"
	# imdb_test_path = "./datasets/imdb/test.tsv"
	#
	# print("Started tokenization.")
	#
	# train_tokenized, test_tokenized, vocab_size, tokenizer = import_clean_tokenize_data(imdb_train_path, imdb_test_path)
	#
	# print("Tokenization complete.")
	#
	# x_train, x_valid, y_train, y_valid, x_test = pad_split_data(train_tokenized, test_tokenized)
	#
	# model, all_preds = load_and_make_predictions_batch(lstm_size, multiple_fc, fc_units, vocab_size, checkpoint_to_restore, x_test)
	#
	# print(all_preds)



########################################################################## LOAD, PREDICT AND GET GRADIENT ATTRIBUTIONS DICTIONARY

	embed_size = 300
	batch_size = 1 #default was 250 for training

	num_layers = 1
	dropout = 0.5
	learning_rate = 0.001
	epochs = 1

	#stick to these parameters while training, restoring and predicting
	lstm_size = 64
	multiple_fc = False
	fc_units = 128

	vocab_size = 99426

	checkpoint_to_restore = "./models/LSTM_models/64,False,128/64,False,128.ckpt"

	# checkpoint_to_create= "/Users/sharan/Desktop/RNN_with_embed/64,False,128.ckpt"


	imdb_train_path = "./datasets/imdb/train.tsv"
	imdb_test_path = "./datasets/imdb/test.tsv"


	tokenizer = load_tokenizer('./tokenizers/tokenizer_imdb.pickle')

	test_data, df = create_test_example(tokenizer) #creates embeddings from single_test_data

	model, all_preds = load_and_make_predictions_single(lstm_size, multiple_fc, fc_units, vocab_size, checkpoint_to_restore, test_data)

	grads = get_gradients(model, all_preds, test_data)
	# print("computed gradients tensor: {}".format(grads))

	grads_list, indices_list =  get_gradients_values(grads)

	text_list = get_word_from_index(indices_list, tokenizer)
	word_list = [e for e in text_list[0].split(" ")]

	# print("indices list: {}".format(indices_list))
	print("words from indices: {}".format(text_list))

	attributions = compress_gradients(grads_list)

	attributions_dict = clean_attributes(attributions, word_list)

	print(attributions_dict)



########################################################################## TRAIN NEW MODEL

	# train_and_checkpoint(checkpoint_to_create, lstm_size, multiple_fc, fc_units, vocab_size)

























