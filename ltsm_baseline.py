import pandas as pd
import numpy as np
import tensorflow as tf
import inspect
import nltk, re, time
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

#GLOBAL VARIABLES

embed_size = 300
batch_size = 250
lstm_size = 128
num_layers = 1
dropout = 0.5
learning_rate = 0.001
epochs = 100
multiple_fc = False
fc_units = 256

def summarize_variable(var):

	print("\n var is of type {} ".format(type(var)))
	print(" var has leading length {} \n".format(len(var)))

	if(type(var) == "<class 'numpy.ndarray'>"):
		print("Shape of numpy array is {} \n".format(var, var.shape))



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

def get_gradients(checkpoint, x_test):

	# this function takes the model, sample input sentence as input, computes the ouput and finds the gradients of each input word

	model = load_model(checkpoint, no_of_unique_words)

	optimizer_here = model.gradients

	cost_here = model.cost

	gradients, variables = zip(*optimizer_here.compute_gradients(cost_here))

	print("grads and vars: {} and {}".format(gradients, variables))

	opt = optimizer_here.apply_gradients(list(zip(gradients, variables)))


	print("opt is {}".format(opt))

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	test_state = sess.run(model.initial_state)

	predicted_y = predict(checkpoint, x_test)

	feed = {model.inputs: x_test,
			model.labels: predicted_y[:, None], #coverting 1d to 2d array
			model.keep_prob: dropout,
			model.initial_state: test_state}
	sess.run(opt, feed_dict=feed)


def build_rnn(n_words, embed_size, batch_size, lstm_size, num_layers,
			  dropout, learning_rate, multiple_fc, fc_units):
	'''Build the Recurrent Neural Network'''


	global opt

	tf.reset_default_graph()

	# Declare placeholders we'll feed into the graph
	with tf.name_scope('inputs'):
		inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

	with tf.name_scope('labels'):
		labels = tf.placeholder(tf.int32, [None, None], name='labels')

	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	# Create the embeddings
	with tf.name_scope("embeddings"):
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
	export_nodes = ['inputs', 'labels', 'keep_prob', 'initial_state', 'final_state', 'accuracy',
					'predictions', 'cost', 'optimizer', 'gradients','merged']
	Graph = namedtuple('Graph', export_nodes)
	local_dict = locals()
	graph = Graph(*[local_dict[each] for each in export_nodes])

	return graph



def train_model(model, epochs, log_string):
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
				if stop_early == 3:
					break

				# Reset stop_early if the validation loss finds a new low
			# Save a checkpoint of the model
			else:
				print("New Record!")
				stop_early = 0
				checkpoint = "/Users/sharan/Desktop/results/sentiment_{}.ckpt".format(
					log_string)
				saver.save(sess, checkpoint)






def predict(checkpoint, x_test):
	'''Predict the sentiment of the testing data'''


	summarize_variable(x_test)

	x_test_pruned = x_test[0:250]

	predicted_y = []

	model = build_rnn(n_words=no_of_unique_words,
					  embed_size=embed_size,
					  batch_size=batch_size,
					  lstm_size=lstm_size,
					  num_layers=num_layers,
					  dropout=dropout,
					  learning_rate=learning_rate,
					  multiple_fc=multiple_fc,
					  fc_units=fc_units)

	with tf.Session() as sess:
		saver = tf.train.Saver()
		# Load the model
		saver.restore(sess, checkpoint)
		test_state = sess.run(model.initial_state)
		print("loaded model and test state")

		for _, x in enumerate(get_test_batches(x_test_pruned, batch_size), 1):

			print("predicting for input {}".format(x))
			feed = {model.inputs: x,
					model.keep_prob: 1,
					model.initial_state: test_state}
			predictions = sess.run(model.predictions, feed_dict=feed)
			for pred in predictions:
				predicted_y.append(float(pred))


	summarize_variable(predicted_y) #all_preds is of type list with same number of rows as x_test_pruned

	for p in predicted_y:
		print("prediction is :{}".format(p))


	return np.array(predicted_y)


def write_submission(predictions, string):

	global train, test

	submission = pd.DataFrame(data={"id": test["id"], "sentiment": predictions})
	submission.to_csv("submission_{}.csv".format(string), index=False, quoting=3)



def load_model(path, no_of_unique_words):

	model = build_rnn(n_words=no_of_unique_words, embed_size=embed_size, batch_size=batch_size, lstm_size=lstm_size,
					  num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, multiple_fc=multiple_fc,
					  fc_units=fc_units)

	saver = tf.train.Saver()

	with tf.Session() as sess:

		print("Restoring model from path {}".format(path_to_restore))
		saver.restore(sess, path)

		print("Model restored and ready to use.")

	return model



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

	no_of_unique_words = len(word_index) # we need this explicit parameter to build the model

	return  train_seq, test_seq, no_of_unique_words



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


def train_and_checkpoint():

	choice = input("Are you sure you want to train models again and overwrite cptk files and checkpoint? 1 for yes, 0 for no")

	if(choice == 1):

		for lstm_size in [64, 128]:
			for multiple_fc in [True, False]:
				for fc_units in [128, 256]:

					log_string = 'ru={},fcl={},fcu={}'.format(lstm_size, multiple_fc, fc_units)
					model = build_rnn(n_words=no_of_unique_words, embed_size=embed_size, batch_size=batch_size, lstm_size=lstm_size, num_layers=num_layers,
									  dropout=dropout, learning_rate=learning_rate, multiple_fc=multiple_fc, fc_units=fc_units)

					train_model(model, epochs, log_string)

		# these are the best models
		checkpoint1 = "/Users/sharan/Desktop/results/sentiment_ru=128,fcl=False,fcu=256.ckpt"
		checkpoint2 = "/Users/sharan/Desktop/results/sentiment_ru=128,fcl=False,fcu=128.ckpt"
		checkpoint3 = "/Users/sharan/Desktop/results/sentiment_ru=64,fcl=True,fcu=256.ckpt"

		# make predictions with these 3 models
		predictions1 = make_predictions(128, False, 256, checkpoint1)
		predictions2 = make_predictions(128, False, 128, checkpoint2)
		predictions3 = make_predictions(64, True, 256, checkpoint3)

		#
		predictions_combined = (pd.DataFrame(predictions1) + pd.DataFrame(predictions2) + pd.DataFrame(predictions3)) / 3

		write_submission(predictions1, "ru=128,fcl=False,fcu=256")
		write_submission(predictions2, "ru=128,fcl=False,fcu=128")
		write_submission(predictions3, "ru=64,fcl=True,fcu=256")
		write_submission(predictions_combined.ix[:, 0], "combined")

	else:
		print("Exitting from Train and checkpoint func.")




if __name__ == '__main__':

	path_to_restore = "/Users/sharan/Desktop/results/sentiment_ru=128,fcl=False,fcu=256.ckpt"

	imdb_train_path = "./imdb/train.tsv"
	imdb_test_path = "./imdb/test.tsv"

	train_tokenized, test_tokenized, no_of_unique_words = import_clean_tokenize_data(imdb_train_path, imdb_test_path)

	x_train, x_valid, y_train, y_valid, x_test = pad_split_data(train_tokenized, test_tokenized)

	model = load_model(path_to_restore, no_of_unique_words)

	#plzz work
	# get_gradients(model)

	#Uncomment below line to train the models and predict, compare best models and see scores
	# train_and_checkpoint()

	checkpoint = "/Users/sharan/Desktop/results/sentiment_ru=128,fcl=False,fcu=256.ckpt"

	get_gradients(checkpoint, x_test[0:250])

	# predict(x_test, model)













