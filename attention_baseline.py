from attention_keras.layers.attention import AttentionLayer
from keras import Input


batch_size = 250


attn = AttentionLayer()



print(locals())



encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize), name='decoder_inputs')


