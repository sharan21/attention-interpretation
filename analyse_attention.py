from attention_baseline import *
from nltk.translate.bleu_score import sentence_bleu

def get_bleu_score(hyp, ref):
	return sentence_bleu(ref, hyp)






if __name__ == '__main__':

	"""Hyperparameters"""
	batch_size = 64
	hidden_size = 96
	en_timesteps, fr_timesteps = 20, 20

	# Use only full size of english and french vocab
	en_vsize = 201
	fr_vsize = 345

	"Model building and Loading"

	params_wrapper_ortho = (batch_size, hidden_size, en_timesteps, fr_timesteps, en_vsize, fr_vsize, True)
	params_wrapper_non_ortho = (batch_size, hidden_size, en_timesteps, fr_timesteps, en_vsize, fr_vsize, False)

	pathtononortho = './models/attention_models/nmt_models/100000_15_nonortho.h5'
	pathtoortho = './models/attention_models/nmt_models/100000_15_ortho.h5'

	ortho_model, infer_enc_model_ortho, infer_dec_model_ortho = load_model_from_params(params_wrapper_ortho, pathtoortho)
	non_ortho_model, infer_enc_model_non_ortho, infer_dec_model_non_ortho = load_model_from_params(params_wrapper_non_ortho, pathtononortho)

	""" Loading tokenizers """
	en_tokenizer = load_tokenizer(path='./tokenizers/en_tokenizer.pickle')
	fr_tokenizer = load_tokenizer(path='./tokenizers/fr_tokenizer.pickle')

	"""Get Reverse dictionary"""

	en_index2word = dict(zip(en_tokenizer.word_index.values(), en_tokenizer.word_index.keys()))
	fr_index2word = dict(zip(fr_tokenizer.word_index.values(), fr_tokenizer.word_index.keys()))

	"""Getting Test sentences"""

	tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data() # train default is 100000

	sample_no = 2

	test_en = ts_en_text[sample_no]
	test_fr = ts_fr_text[sample_no]

	test_en_seq = sents2sequences(en_tokenizer, [test_en], pad_length=en_timesteps)
	test_fr_seq = sents2sequences(fr_tokenizer, [test_fr], pad_length=fr_timesteps)
	# print(ts_en_text)



	# """Generating predictions"""
	#
	# logger.info('Translating: {}\n'.format(test_en))
	#
	# test_fr_non_ortho, attn_weights_non_ortho = infer_nmt(
	# 	encoder_model=infer_enc_model_non_ortho, decoder_model=infer_dec_model_non_ortho,
	# 	test_en_seq=test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer,
	# 	fr_index2word=fr_index2word)
	#
	# test_fr_ortho, attn_weights_ortho = infer_nmt(
	# 	encoder_model=infer_enc_model_ortho, decoder_model=infer_dec_model_ortho,
	# 	test_en_seq=test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer,
	# 	fr_index2word=fr_index2word)
	#
	# logger.info('\t ORTHO says French is: {}\n'.format(test_fr_ortho))
	# logger.info('\t NON-ORTHO says French is: {}\n'.format(test_fr_non_ortho))

	"""Get Bleu Scores for ortho"""
	overall_bleu_score = 0
	sum = 0
	assert(len(ts_en_text) == len(ts_fr_text))
	test_count = 10

	for i in range(test_count):
		test = ts_en_text[i]

		test_seq = sents2sequences(en_tokenizer, [test_en], pad_length=en_timesteps)


		hyp, _ = infer_nmt(
			encoder_model=infer_enc_model_non_ortho, decoder_model=infer_dec_model_non_ortho,
			test_en_seq=test_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer,
			fr_index2word=fr_index2word)


		bs = get_bleu_score(hyp, ts_fr_text[i])
		sum += bs

		print("hyp is : {}".format(hyp))
		print("red is : {}".format(ts_fr_text[i]))
		print("bs is : {}\n".format(bs))

	overall_bleu_score = float(sum/test_count)
	print("Overall BS : {}\n".format(overall_bleu_score))
























