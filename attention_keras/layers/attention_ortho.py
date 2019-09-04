import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import numpy as np
from tensorflow.python.keras import backend as K

class AttentionLayerOrtho(Layer):

	def __init__(self, **kwargs):
		super(AttentionLayerOrtho, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		# Create a trainable weight variable for this layer.

		self.W_a = self.add_weight(name='W_a',
								   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
								   initializer='uniform',
								   trainable=True)
		self.U_a = self.add_weight(name='U_a',
								   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
								   initializer='uniform',
								   trainable=True)
		self.V_a = self.add_weight(name='V_a',
								   shape=tf.TensorShape((input_shape[0][2], 1)),
								   initializer='uniform',
								   trainable=True)

		super(AttentionLayerOrtho, self).build(input_shape)  # Be sure to call this at the end

	def call(self, inputs, verbose=False):
		"""
		        inputs: [encoder_output_sequence, decoder_output_sequence]
		        """
		assert type(inputs) == list
		encoder_out_seq, decoder_out_seq = inputs

		if verbose:
			print('encoder_out_seq>', encoder_out_seq.shape)
			print('decoder_out_seq>', decoder_out_seq.shape)


		# CONVERT ENCODER_INPUTS TO ORTHOGONAL


		# UNCOMMENT THE BELOW LINE FOR BATCH_SIZE = 1
		encoder_out_seq = self.orthogonalize_encoder_inputs(encoder_out_seq)
		# encoder_out_seq = self.orthogonalize_encoder_inputs_with_batch(encoder_out_seq)

		def energy_step(inputs, states):
			""" Step function for computing energy for a single decoder state """

			assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
			assert isinstance(states, list) or isinstance(states, tuple), assert_msg

			""" Some parameters required for shaping tensors"""
			en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
			de_hidden = inputs.shape[-1]

			""" Computing S.Wa where S=[s0, s1, ..., si]"""
			# <= batch_size*en_seq_len, latent_dim
			reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
			# <= batch_size*en_seq_len, latent_dim
			W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
			if verbose:
				print('wa.s>', W_a_dot_s.shape)

			""" Computing hj.Ua """
			U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
			if verbose:
				print('Ua.h>', U_a_dot_h.shape)

			""" tanh(S.Wa + hj.Ua) """
			# <= batch_size*en_seq_len, latent_dim
			reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
			if verbose:
				print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

			""" softmax(va.tanh(S.Wa + hj.Ua)) """
			# <= batch_size, en_seq_len
			e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
			# <= batch_size, en_seq_len
			e_i = K.softmax(e_i)

			if verbose:
				print('ei>', e_i.shape)

			return e_i, [e_i]

		def context_step(inputs, states):
			""" Step function for computing ci using ei """
			# <= batch_size, hidden_size
			c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
			if verbose:
				print('ci>', c_i.shape)
			return c_i, [c_i]

		def create_inital_state(inputs, hidden_size):
			# We are not using initial states, but need to pass something to K.rnn funciton
			fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
			fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
			fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
			fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
			return fake_state

		fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
		fake_state_e = create_inital_state(encoder_out_seq,
										   encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

		""" Computing energy outputs """
		# e_outputs => (batch_size, de_seq_len, en_seq_len)
		last_out, e_outputs, _ = K.rnn(
			energy_step, decoder_out_seq, [fake_state_e],
		)

		""" Computing context vectors """
		last_out, c_outputs, _ = K.rnn(
			context_step, e_outputs, [fake_state_c],
		)

		return c_outputs, e_outputs

	def compute_output_shape(self, input_shape):

		return [
			tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
			tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
		]

	def orthogonalize_encoder_inputs(self, encoder_out_seq):

		# Number of runs = Number of words = Indices = 20
		runs = encoder_out_seq[0].shape._dims[0]
		#reshape the encoder tensor to change leading dim.
		encoder_out_seq_reshaped = tf.transpose(encoder_out_seq)
		#convert reshaped tensor into variable to use scatter_update()
		encoder_out_seq_reshaped_v = tf.Variable(encoder_out_seq_reshaped)


		for i in range(runs):
			for j in range(runs):
				# indices to update the new orthogonalized variable using scatter_update
				indices = tf.convert_to_tensor(np.array([i], dtype=np.int32))
				#Stop once you reach yourself
				if(i == j):
					break
				num = tf.multiply(encoder_out_seq_reshaped_v[i], encoder_out_seq_reshaped_v[j])
				den = tf.multiply(encoder_out_seq_reshaped_v[i], encoder_out_seq_reshaped_v[i])
				angle = tf.divide(num, den)
				cos_factor = tf.multiply(angle, encoder_out_seq_reshaped_v[i])
				#reshape needed to use scatter_update()
				cos_factor_reshaped = tf.expand_dims(cos_factor, 0)
				# equivalent: h_tilda = h - proj
				tf.scatter_update(encoder_out_seq_reshaped_v, indices, encoder_out_seq_reshaped_v[i] - cos_factor_reshaped)

		#encoder inputs variable now orthogonalised
		#extract back tensor from the variable

		encoder_out_seq_ortho = encoder_out_seq_reshaped_v._initial_value
		encoder_out_seq_final = tf.transpose(encoder_out_seq_reshaped_v)

		return encoder_out_seq_final

	def orthogonalize_encoder_inputs_with_batch(self, encoder_out_seq):

		# Number of runs = Number of words = Indices = 20
		runs = int(encoder_out_seq.shape[1])
		# To modify the tensor it first must be converted into a variable

		# encoder_out_seq_reshaped = tf.reshape(encoder_out_seq, (20, 64, 96))
		# encoder_out_seq_reshaped = tf.transpose(encoder_out_seq, [1, 0, 2])
		encoder_out_seq_reshaped = tf.transpose(encoder_out_seq)
		encoder_out_seq_reshaped_v = tf.Variable(encoder_out_seq_reshaped)

		for i in range(runs):
			for j in range(runs):
				indices = tf.convert_to_tensor(np.array([i], dtype=np.int32))
				# Stop once you reach yourself
				if (i == j):
					break
				num = tf.multiply(encoder_out_seq_reshaped_v[i], encoder_out_seq_reshaped_v[j])
				den = tf.multiply(encoder_out_seq_reshaped_v[i], encoder_out_seq_reshaped_v[i])
				angle = tf.divide(num, den)
				cos_factor = tf.multiply(angle, encoder_out_seq_reshaped_v[i])
				# cos_factor_reshaped = tf.reshape(cos_factor, (1, 64, 96))
				cos_factor_reshaped = tf.expand_dims(cos_factor, 0)


				tf.scatter_update(encoder_out_seq_reshaped_v, indices, encoder_out_seq_reshaped_v[i] - cos_factor_reshaped)

		# encoder_out_seq_ortho = tf.reshape(encoder_out_seq_reshaped_v, (64, 20, 96))
		# encoder_out_seq_ortho = tf.transpose(encoder_out_seq_reshaped_v, [1, 0, 2])
		encoder_out_seq_ortho = tf.transpose(encoder_out_seq_reshaped_v)

		return(encoder_out_seq_ortho)













