import rnn_eeg_ad as rnn
import sys

# Note: we create global variables in the rnn module, because functions there expect that

rnn.dense1 = 0
rnn.lstm1 = 8
rnn.lstm2 = 8
rnn.lstm3 = 0

file_id = ''
if len(sys.argv) > 1: file_id = sys.argv[1]

rnn.subjs_train_perm = ( (tuple(i for i in range(2, 15)) + tuple(i for i in range(18, 35)), ()), )
rnn.subjs_test = (0, 1, 15, 16, 17)

rnn.decimation = 0
rnn.epochs = 20
rnn.oversample = True
rnn.pca = True
rnn.window = 256
rnn.overlap = rnn.window // 2
rnn.multiscale_pca = False

for rnn.spikes in (1/1000, 1/500, 1/200, 1/100):
	for rnn.rpca in (False,):
		if rnn.decimation:
			rnn.window //= rnn.decimation
			rnn.overlap //= rnn.decimation
		rnn.x_data, rnn.y_data, rnn.subj_inputs = rnn.create_dataset(rnn.window, rnn.overlap, rnn.decimation)
		rnn.train_session(save_model = False, earlystop = 0, train_split = 0.75, file_id = file_id)
