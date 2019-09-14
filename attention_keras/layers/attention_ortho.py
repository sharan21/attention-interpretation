import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import numpy as np
import keras
from tensorflow.python.keras import backend as K

class AttentionLayerOrtho(Layer):

	def __init__(self, **kwargs):
		super(AttentionLayerOrtho, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		# Create a trainable weight variable for this layer.

		w_init = tf.random_normal_initializer()

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

		self.ortho_states = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False,name="ortho_states")
		# self.ortho_states_temp = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="ortho_states_temp")
		self.buffer = tf.Variable([], validate_shape=False, dtype=tf.float32, trainable=False, name="buffer")



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
		# encoder_out_seq = self.orthogonalize_encoder_inputs(encoder_out_seq)

		encoder_out_seq_list = self.orthogonalize_encoder_inputs_new_new_new(encoder_out_seq)
		encoder_out_seq_t = tf.concat(encoder_out_seq_list, 0)
		encoder_out_seq = tf.transpose(encoder_out_seq_t, [1, 0, 2])




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


	def orthogonalize_encoder_inputs(self, encoder_out_seq): # can be used for both stochiastic and batch

		runs = int(encoder_out_seq.shape[1])

		encoder_out_seq_reshaped = tf.transpose(encoder_out_seq, [1, 0, 2])
		encoder_out_seq_reshaped_v = tf.Variable(encoder_out_seq_reshaped)

		for i in range(runs):

			for j in range(runs):

				indices = tf.convert_to_tensor(np.array([i], dtype=np.int32))
				if (i == j):
					break

				num = tf.multiply(encoder_out_seq_reshaped_v[i], encoder_out_seq_reshaped_v[j])
				den = tf.multiply(encoder_out_seq_reshaped_v[i], encoder_out_seq_reshaped_v[i])
				angle = tf.divide(num, den)
				cos_factor = tf.multiply(angle, encoder_out_seq_reshaped_v[i])
				cos_factor_reshaped = tf.expand_dims(cos_factor, 0)

				encoder_out_seq_reshaped_v = tf.scatter_update(encoder_out_seq_reshaped_v, indices,
								  encoder_out_seq_reshaped_v[i] - cos_factor_reshaped)


		encoder_out_seq_ortho = encoder_out_seq_reshaped_v._initial_value
		encoder_out_seq_final = tf.transpose(encoder_out_seq_ortho, [1, 0, 2])

		return (encoder_out_seq_final)

	def orthogonalize_encoder_inputs_new(self, encoder_out_seq):  # can be used for both stochastic and batch


		runs = int(encoder_out_seq.shape[1])

		encoder_out_seq_reshaped = tf.transpose(encoder_out_seq, [1, 0, 2])
		self.ortho_states = tf.assign(self.ortho_states, encoder_out_seq_reshaped, validate_shape=False)
		encoder_out_seq_reshaped_v = self.ortho_states

		for i in range(runs):

			for j in range(runs):

				indices = tf.convert_to_tensor(np.array([i], dtype=np.int32))
				if (i == j):
					break

				num = tf.multiply(encoder_out_seq_reshaped_v[i], encoder_out_seq_reshaped_v[j])
				den = tf.multiply(encoder_out_seq_reshaped_v[i], encoder_out_seq_reshaped_v[i])
				angle = tf.divide(num, den)
				cos_factor = tf.multiply(angle, encoder_out_seq_reshaped_v[i])
				cos_factor_reshaped = tf.expand_dims(cos_factor, 0)


				encoder_out_seq_reshaped_v = tf.scatter_update(encoder_out_seq_reshaped_v, indices,
				                                               encoder_out_seq_reshaped_v[i] - cos_factor_reshaped,
				                                               use_locking=True)
				tf.stop_gradient(
					encoder_out_seq_reshaped_v,
					name="ScatterUpdate_189"
				)


		return (tf.transpose(encoder_out_seq_reshaped_v, [1, 0, 2]))


	def orthogonalize_encoder_inputs_new_new(self, encoder_out_seq):  # can be used for both stochastic and batch

		t_list = []

		runs = int(encoder_out_seq.shape[1])

		encoder_out_seq_reshaped = tf.transpose(encoder_out_seq, [1, 0, 2])

		self.ortho_states_temp = tf.assign(self.ortho_states, encoder_out_seq_reshaped, validate_shape=False)

		for i in range(runs):

			self.buffer = tf.assign(self.buffer, self.ortho_states_temp[i], validate_shape=False)
			for j in range(runs):

				if (i == j):
					break

				num = tf.multiply(self.ortho_states_temp[i], self.ortho_states_temp[j])
				den = tf.multiply(self.ortho_states_temp[i], self.ortho_states_temp[i])
				angle = tf.divide(num, den)
				cos_factor = tf.multiply(angle, self.ortho_states_temp[i])
				# cos_factor_reshaped = tf.expand_dims(cos_factor, 0)

				self.buffer = tf.assign(self.buffer, self.buffer-cos_factor)

			t_list.append(tf.expand_dims(self.buffer, 0))
			# self.final_states = self.final_states[i].assign(tf.expand_dims(self.buffer, 0))

		return(t_list)


	def orthogonalize_encoder_inputs_new_new_new(self, encoder_out_seq):  # can be used for both stochastic and batch

		t_list = []

		runs = int(encoder_out_seq.shape[1])

		encoder_out_seq_reshaped = tf.transpose(encoder_out_seq, [1, 0, 2])

		self.ortho_states = tf.assign(self.ortho_states, encoder_out_seq_reshaped, validate_shape=False)

		for i in range(runs):

			self.buffer = tf.assign(self.buffer, self.ortho_states[i], validate_shape=False)
			for j in range(runs):

				if (i == j):
					break

				num = tf.multiply(self.ortho_states[i], self.ortho_states[j])
				den = tf.multiply(self.ortho_states[i], self.ortho_states[i])
				angle = tf.divide(num, den)
				cos_factor = tf.multiply(angle, self.ortho_states[i])
				# cos_factor_reshaped = tf.expand_dims(cos_factor, 0)

				self.buffer = tf.assign(self.buffer, self.buffer-cos_factor)

			t_list.append(tf.expand_dims(self.buffer, 0))
			# self.final_states = self.final_states[i].assign(tf.expand_dims(self.buffer, 0))

		return(t_list)

		# return (tf.transpose(self.ortho_states, [1, 0, 2]))

