from attention_baseline import *
import pickle

def save_tokenizer(tokenizer, path='./tokenizers/unnames_tokenizer.pickle'):

	print("saving tokenizer...")
	with open(path, 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(path = './tokenizers/tokenizer_imdb.pickle'):

	print("loading tokenizer...")
	with open(path, 'rb') as handle:
		tokenizer = pickle.load(handle)

	return tokenizer


if __name__ == "__main__":
	# TODO Need to train models with full en_vsize and fr_vsize


	train_size = 100000

	tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data(train_size=train_size)

	""" Defining tokenizers """
	en_tokenizer = load_tokenizer(path='./tokenizers/en_tokenizer.pickle')
	fr_tokenizer = load_tokenizer(path='./tokenizers/fr_tokenizer.pickle')



	""" Getting preprocessed data """
	en_seq, fr_seq = preprocess_data(en_tokenizer, fr_tokenizer, tr_en_text, tr_fr_text, en_timesteps, fr_timesteps)


	# Full Vocab Size for MT Dataset, Use only these for training
	en_vsize = 201
	fr_vsize = 345


	""" Load Model"""

	model, infer_enc_model, infer_dec_model = load_attn_model('./models/attention_models/nmt_models/nmt_100000_10.h5')


	""" Index2word """
	en_index2word = dict(zip(en_tokenizer.word_index.values(), en_tokenizer.word_index.keys()))
	fr_index2word = dict(zip(fr_tokenizer.word_index.values(), fr_tokenizer.word_index.keys()))

	""" Inferring with trained model """

	sample = 4

	test_en = ts_en_text[sample]
	test_fr = ts_fr_text[sample]
	custom_input = 'Hello my name is sharan, and I am from India \n'

	logger.info('Translating: {}'.format(test_en))

	test_en_seq = sents2sequences(en_tokenizer, [test_en], pad_length=en_timesteps)
	test_fr_seq = sents2sequences(fr_tokenizer, [test_fr], pad_length=fr_timesteps)
	custom_input_seq = sents2sequences(en_tokenizer, [custom_input], pad_length=en_timesteps)

	test_fr, attn_weights = infer_nmt(
		encoder_model=infer_enc_model, decoder_model=infer_dec_model,
		test_en_seq=test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer, fr_index2word=fr_index2word)

	logger.info('\tFrench: {}'.format(test_fr))

