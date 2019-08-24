import tensorflow.keras as keras
from random import sample

from tensorflow.python.keras.utils import to_categorical
import numpy as np
import os, sys

project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
if project_path not in sys.path:
    sys.path.append(project_path)

from attention_keras.examples.utils.data_helper import read_data, sents2sequences
from attention_keras.examples.nmt.model import define_nmt
from attention_keras.examples.utils.model_helper import plot_attention_weights
from attention_keras.examples.utils.logger import get_logger

base_dir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
logger = get_logger("examples.nmt.train",os.path.join(base_dir, 'logs'))

batch_size = 64
hidden_size = 96

# en_timesteps, fr_timesteps = 20, 20 # for unmasked inputs

en_timesteps, fr_timesteps = 20, 20 # for masked inputs

def get_data(train_size, random_seed=100):

    """ Getting randomly shuffled training / testing data """
    en_text = read_data(os.path.join(project_path, 'data', 'small_vocab_en.txt'))
    fr_text = read_data(os.path.join(project_path, 'data', 'small_vocab_fr.txt'))
    logger.info('Length of text: {}'.format(len(en_text)))

    fr_text = ['sos ' + sent[:-1] + 'eos .'  if sent.endswith('.') else 'sos ' + sent + ' eos .' for sent in fr_text]

    np.random.seed(random_seed)
    inds = np.arange(len(en_text))
    np.random.shuffle(inds)

    train_inds = inds[:train_size]
    test_inds = inds[train_size:]
    tr_en_text = [en_text[ti] for ti in train_inds]
    tr_fr_text = [fr_text[ti] for ti in train_inds]

    ts_en_text = [en_text[ti] for ti in test_inds]
    ts_fr_text = [fr_text[ti] for ti in test_inds]

    return tr_en_text, tr_fr_text, ts_en_text, ts_fr_text

def find_start_word(seq_text):
    '''takes 1d tokenized vector and return start index'''
    for i in range(len(seq_text)):
        if(seq_text[i] != 0):
            return i

def add_intermediate_padding(en_seq, fr_seq = [], pad_size = 5):
    '''Adds 0s in between the sentence or randomly assorted'''
    en_seq_padded = []
    fr_seq_padded = []

    #Refactor code to fuse both loops
    for e in en_seq:
        start_index = find_start_word(e)
        end_index = len(e)
        mid = int((start_index+end_index)/2)

        zeros = np.zeros(pad_size, dtype=np.int32)
        new_e = np.concatenate((e[0:mid], zeros, e[mid:]))
        en_seq_padded.append(new_e)

    for f in fr_seq:
        start_index = find_start_word(f)
        end_index = len(f)
        mid = int((start_index+end_index)/2)

        zeros = np.zeros(pad_size, dtype=np.int32)
        new_f = np.concatenate((f[0:mid], zeros, f[mid:]))
        fr_seq_padded.append(new_f)

    return np.array(en_seq_padded), np.array(fr_seq_padded)

def add_random_padding(en_seq, fr_seq = []):
    '''Adds 2, 0 padding to all the train sentences'''

    en_seq_padded = []
    fr_seq_padded = []

    # Refactor code to fuse both loops
    for e in en_seq:
        start_index = find_start_word(e)
        end_index = len(e)
        no_of_words = end_index-start_index

        # no_of_random_indices = int(0.2*no_of_words) # for 20 percent of the words in the test sentence
        no_of_random_indices = 2
        index_list = np.arange(start_index, end_index)
        random_indices = sample(list(index_list), no_of_random_indices)
        new_e = e

        for r in random_indices:
            new_e = np.insert(new_e, r+1, 0)

        en_seq_padded.append(new_e)

    for f in fr_seq:
        start_index = find_start_word(f)
        end_index = len(f)
        no_of_words = end_index - start_index

        # no_of_random_indices = int(0.2*no_of_words) # for 20 percent of the words in the test sentence
        no_of_random_indices = 2
        index_list = np.arange(start_index, end_index)
        random_indices = sample(list(index_list), no_of_random_indices)
        new_f = f

        for r in random_indices:
            new_f = np.insert(new_f, r + 1, 0)

        fr_seq_padded.append(new_f)

    return np.array(en_seq_padded), np.array(fr_seq_padded)

def save_model(model, pathtojson = "./nmt_models/untitled.json", pathtoh5 = "./nmt_models/untitled.h5"):

    print("Saving model")

    model_json = model.to_json()
    with open(pathtojson, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(pathtoh5)

    print("Saved model to disk!")




def preprocess_data(en_tokenizer, fr_tokenizer, en_text, fr_text, en_timesteps, fr_timesteps):
    """ Preprocessing data and getting a sequence of word indices """

    en_seq = sents2sequences(en_tokenizer, en_text, reverse=False, padding_type='pre', pad_length=en_timesteps)
    fr_seq = sents2sequences(fr_tokenizer, fr_text, pad_length=fr_timesteps)
    logger.info('Vocabulary size (English): {}'.format(np.max(en_seq)+1))
    logger.info('Vocabulary size (French): {}'.format(np.max(fr_seq)+1))
    logger.debug('En text shape: {}'.format(en_seq.shape))
    logger.debug('Fr text shape: {}'.format(fr_seq.shape))

    return en_seq, fr_seq


def train(full_model, en_seq, fr_seq, batch_size, n_epochs=10):
    """ Training the model """

    for ep in range(n_epochs):
        losses = []
        for bi in range(0, en_seq.shape[0] - batch_size, batch_size):

            en_onehot_seq = to_categorical(en_seq[bi:bi + batch_size, :], num_classes=en_vsize)
            fr_onehot_seq = to_categorical(fr_seq[bi:bi + batch_size, :], num_classes=fr_vsize)

            full_model.train_on_batch([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :])

            l = full_model.evaluate([en_onehot_seq, fr_onehot_seq[:, :-1, :]], fr_onehot_seq[:, 1:, :],
                                    batch_size=batch_size, verbose=0)

            losses.append(l)
        if (ep + 1) % 1 == 0:
            logger.info("Loss in epoch {}: {}".format(ep + 1, np.mean(losses)))


def infer_nmt(encoder_model, decoder_model, test_en_seq, en_vsize, fr_vsize):
    """
    Infer logic
    :param encoder_model: keras.Model
    :param decoder_model: keras.Model
    :param test_en_seq: sequence of word ids
    :param en_vsize: int
    :param fr_vsize: int
    :return:
    """

    test_fr_seq = sents2sequences(fr_tokenizer, ['sos'], fr_vsize)
    test_en_onehot_seq = to_categorical(test_en_seq, num_classes=en_vsize)
    test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

    enc_outs, enc_last_state = encoder_model.predict(test_en_onehot_seq)
    dec_state = enc_last_state
    attention_weights = []
    fr_text = ''

    for i in range(20):

        dec_out, attention, dec_state = decoder_model.predict([enc_outs, dec_state, test_fr_onehot_seq])
        dec_ind = np.argmax(dec_out, axis=-1)[0, 0]

        if dec_ind == 0:
            break
        test_fr_seq = sents2sequences(fr_tokenizer, [fr_index2word[dec_ind]], fr_vsize)
        test_fr_onehot_seq = np.expand_dims(to_categorical(test_fr_seq, num_classes=fr_vsize), 1)

        attention_weights.append((dec_ind, attention))
        fr_text += fr_index2word[dec_ind] + ' '

    return fr_text, attention_weights


if __name__ == '__main__':

    pass

    ############################################################ EXPERIMENTAL SECTION 1 FOR RANDOM 0 PADDING

    # debug = True
    # """ Hyperparameters """
    #
    # train_size = 100000 if not debug else 10000
    # filename = ''
    #
    # tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data(train_size=train_size)
    #
    # """ Defining tokenizers """
    # en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    # en_tokenizer.fit_on_texts(tr_en_text)
    #
    # fr_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    # fr_tokenizer.fit_on_texts(tr_fr_text)
    #
    # """ Getting preprocessed data """
    # en_seq, fr_seq = preprocess_data(en_tokenizer, fr_tokenizer, tr_en_text, tr_fr_text, en_timesteps, fr_timesteps)
    #
    # start_indices = []
    # for e in en_seq:
    #     start_indices.append(find_start_word(e))
    #
    # en_seq_padded, fr_seq_padded = add_random_padding(en_seq, fr_seq)







    ############################################################ EXPERIMENTAL SECTION FOR INTERMEDIATE 0 PADDING

    # debug = True
    # """ Hyperparameters """
    #
    # train_size = 100000 if not debug else 10000
    # filename = ''
    #
    # tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data(train_size=train_size)
    #
    # """ Defining tokenizers """
    # en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    # en_tokenizer.fit_on_texts(tr_en_text)
    #
    # fr_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    # fr_tokenizer.fit_on_texts(tr_fr_text)
    #
    # """ Getting preprocessed data """
    # en_seq, fr_seq = preprocess_data(en_tokenizer, fr_tokenizer, tr_en_text, tr_fr_text, en_timesteps, fr_timesteps)
    #
    # start_indices = []
    # for e in en_seq:
    #     start_indices.append(find_start_word(e))
    #
    # en_seq_padded, fr_seq_padded= add_intermediate_padding(en_seq, fr_seq)
    #
    # en_vsize = max(en_tokenizer.index_word.keys()) + 1
    # fr_vsize = max(fr_tokenizer.index_word.keys()) + 1
    #
    # """ Defining the full model """
    # # below line is for padded inputs of size 5
    # full_model, infer_enc_model, infer_dec_model = define_nmt(
    #     hidden_size=hidden_size, batch_size=batch_size,
    #     en_timesteps=en_timesteps+5, fr_timesteps=fr_timesteps+5,
    #     en_vsize=en_vsize, fr_vsize=fr_vsize)
    #
    #
    # # below line is default model with unpadded inputs
    # # full_model, infer_enc_model, infer_dec_model = define_nmt(
    # #     hidden_size=hidden_size, batch_size=batch_size,
    # #     en_timesteps=en_timesteps, fr_timesteps=fr_timesteps,
    # #     en_vsize=en_vsize, fr_vsize=fr_vsize)
    #
    # n_epochs = 10 if not debug else 3
    #
    # train(full_model, en_seq_padded, fr_seq_padded, batch_size, n_epochs)
    #
    # """ Save model """
    # if not os.path.exists(os.path.join('..', 'h5.models')):
    #     os.mkdir(os.path.join('..', 'h5.models'))
    # full_model.save(os.path.join('..', 'h5.models', 'nmt.h5'))
    #
    # save_model(full_model, pathtojson="./nmt_models/test1.json", pathtoh5="./nmt_models/test1.h5")
    #
    # """ Index2word """
    # en_index2word = dict(zip(en_tokenizer.word_index.values(), en_tokenizer.word_index.keys()))
    # fr_index2word = dict(zip(fr_tokenizer.word_index.values(), fr_tokenizer.word_index.keys()))
    #
    # """ Inferring with trained model """
    # test_en = ts_en_text[0]
    # logger.info('Translating: {}'.format(test_en))
    #
    # test_en_seq = sents2sequences(en_tokenizer, [test_en], pad_length=en_timesteps)
    # test_en_seq_padded, _ = add_intermediate_padding(test_en_seq)
    # test_fr, attn_weights = infer_nmt(
    #     encoder_model=infer_enc_model, decoder_model=infer_dec_model,
    #     test_en_seq=test_en_seq_padded, en_vsize=en_vsize, fr_vsize=fr_vsize)
    # logger.info('\tFrench: {}'.format(test_fr))
    #
    # # """ Attention plotting """
    # # plot_attention_weights(test_en_seq, attn_weights, en_index2word, fr_index2word, base_dir=base_dir)








    ############################################################ MAIN SECTION TO TRAIN DEFAULT NMT MODEL

    debug = False
    """ Hyperparameters """

    train_size = 100000 if not debug else 10000
    filename = ''

    tr_en_text, tr_fr_text, ts_en_text, ts_fr_text = get_data(train_size=train_size)

    """ Defining tokenizers """
    en_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    en_tokenizer.fit_on_texts(tr_en_text)

    fr_tokenizer = keras.preprocessing.text.Tokenizer(oov_token='UNK')
    fr_tokenizer.fit_on_texts(tr_fr_text)

    """ Getting preprocessed data """
    en_seq, fr_seq = preprocess_data(en_tokenizer, fr_tokenizer, tr_en_text, tr_fr_text, en_timesteps, fr_timesteps)

    en_vsize = max(en_tokenizer.index_word.keys()) + 1
    fr_vsize = max(fr_tokenizer.index_word.keys()) + 1

    """ Defining the full model """
    full_model, infer_enc_model, infer_dec_model = define_nmt(
        hidden_size=hidden_size, batch_size=batch_size,
        en_timesteps=en_timesteps, fr_timesteps=fr_timesteps,
        en_vsize=en_vsize, fr_vsize=fr_vsize)

    n_epochs = 10 if not debug else 3
    train(full_model, en_seq, fr_seq, batch_size, n_epochs)

    """ Save model """
    if not os.path.exists(os.path.join('..', 'h5.models')):
        os.mkdir(os.path.join('..', 'h5.models'))
    full_model.save(os.path.join('..', 'h5.models', 'nmt.h5'))

    save_model(full_model, pathtojson="./nmt_models/without_pads.json", pathtoh5="./nmt_models/without_pads.h5")

    """ Index2word """
    en_index2word = dict(zip(en_tokenizer.word_index.values(), en_tokenizer.word_index.keys()))
    fr_index2word = dict(zip(fr_tokenizer.word_index.values(), fr_tokenizer.word_index.keys()))

    """ Inferring with trained model """
    test_en = ts_en_text[0]
    logger.info('Translating: {}'.format(test_en))

    test_en_seq = sents2sequences(en_tokenizer, [test_en], pad_length=en_timesteps)
    test_fr, attn_weights = infer_nmt(
        encoder_model=infer_enc_model, decoder_model=infer_dec_model,
        test_en_seq=test_en_seq, en_vsize=en_vsize, fr_vsize=fr_vsize)
    logger.info('\tFrench: {}'.format(test_fr))

    """ Attention plotting """
    plot_attention_weights(test_en_seq, attn_weights, en_index2word, fr_index2word, base_dir=base_dir)

