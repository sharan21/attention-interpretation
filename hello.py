from build_baseline import build_rnn

embed_size = 300
batch_size = 250
lstm_size = 128
num_layers = 1
dropout = 0.5
learning_rate = 0.001
epochs = 100
multiple_fc = False
fc_units = 256


model = build_rnn(n_words=n_words, embed_size=embed_size, batch_size=batch_size, lstm_size=lstm_size, num_layers=num_layers,
								  dropout=dropout, learning_rate=learning_rate, multiple_fc=multiple_fc, fc_units=fc_units)