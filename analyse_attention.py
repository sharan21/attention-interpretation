from attention_baseline import *
from nltk.translate.bleu_score import sentence_bleu
from googletrans import Translator
from tqdm import tqdm


def get_bleu_score(hyp, ref):
	return sentence_bleu(ref, hyp, weights=(1,0,0,0))

def remove_word_junk(word):
	return ''.join([w for w in word if(w != '.' and w != ',')])

def remove_sentence_junk(sentence):

	words = sentence.split(" ")
	cleaned_words = [remove_word_junk(word) for word in words]
	cleaned_text = ' '.join([word for word in cleaned_words if(word != "eos"
	                                                           and word != "sos" and word != ',' and word != '.' and word != '\n')])

	return cleaned_text


def write_to_file(hyp, ref):
	f = open("./analysis/ig_vs_sg.txt", "a")








if __name__ == '__main__':




	########################################################v##############vv##############v##############v##############vv



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


	"""Generating predictions"""

	logger.info('Translating: {}\n'.format(test_en))

	test_fr_non_ortho, attn_weights_non_ortho = infer_nmt(
		encoder_model=infer_enc_model_non_ortho, decoder_model=infer_dec_model_non_ortho,
		test_en_seq=test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer,
		fr_index2word=fr_index2word)

	test_fr_ortho, attn_weights_ortho = infer_nmt(
		encoder_model=infer_enc_model_ortho, decoder_model=infer_dec_model_ortho,
		test_en_seq=test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer,
		fr_index2word=fr_index2word)

	logger.info('\t ORTHO says French is: {}\n'.format(test_fr_ortho))
	logger.info('\t NON-ORTHO says French is: {}\n'.format(test_fr_non_ortho))

	# """Get Bleu Scores"""
	# t = Translator()
	#
	# overall_bleu_score = 0
	# sum = 0
	# assert(len(ts_en_text) == len(ts_fr_text))
	# test_count = 2000
	#
	#
	#
	# for i in tqdm(range(test_count)):
	#
	# 	test_seq = sents2sequences(en_tokenizer, [ts_en_text[i]], pad_length=en_timesteps)
	#
	#
	# 	# hyp, _ = infer_nmt(
	# 	# 	encoder_model=infer_enc_model_ortho, decoder_model=infer_dec_model_ortho,
	# 	# 	test_en_seq=test_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer,
	# 	# 	fr_index2word=fr_index2word)
	#
	# 	hyp, _ = infer_nmt(
	# 		encoder_model=infer_enc_model_non_ortho, decoder_model=infer_dec_model_non_ortho,
	# 		test_en_seq=test_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer,
	# 		fr_index2word=fr_index2word)
	#
	#
	# 	ref = ts_fr_text[i]
	# 	hyp_cleaned = remove_sentence_junk(hyp)
	# 	ref_cleaned = remove_sentence_junk(ref)
	#
	# 	bs = get_bleu_score(hyp_cleaned, ref_cleaned)
	# 	sum += bs
	# 	# print("en ref is: {}".format(remove_sentence_junk(ts_en_text[i])))
	# 	# print("fr ref is : {}".format(ref_cleaned))
	# 	# # print("google trans of fr ref: {}\n".format(t.translate(ref_cleaned, src='fr', dest='en').text))
	# 	# print("hyp is : {}\n ".format(hyp_cleaned))
	# 	# # print("google trans of hyp: {}\n".format(t.translate(hyp_cleaned, src='fr', dest='en').text))
	# 	#
	# 	# print("bs is : {}\n".format(bs))
	#
	# overall_bleu_score = float(sum/test_count)
	# print("Overall BS : {}\n".format(overall_bleu_score))

	f1 = open("./test_ref.txt", "a")
	f2 = open("./test_hyp_ortho.txt", "a")


	for i in tqdm(range(5000)):

		test_seq = sents2sequences(en_tokenizer, [ts_en_text[i]], pad_length=en_timesteps)

		hyp, _ = infer_nmt(
			encoder_model=infer_enc_model_ortho, decoder_model=infer_dec_model_ortho,
			test_en_seq=test_seq, en_vsize=en_vsize, fr_vsize=fr_vsize, fr_tokenizer=fr_tokenizer,
			fr_index2word=fr_index2word)

		f2.write(remove_sentence_junk(hyp)+"\n")


























