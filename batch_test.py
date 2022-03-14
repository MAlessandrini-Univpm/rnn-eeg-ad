import rnn_eeg_ad as rnn
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

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
rnn.rpca = True
rnn.spikes = 1/500
rnn.rpca_mu = 0.1

for rnn.window in (256, ):
	for rnn.overlap in (rnn.window // 2, ):
		if rnn.decimation:
			rnn.window //= rnn.decimation
			rnn.overlap //= rnn.decimation
		rnn.x_data, rnn.y_data, rnn.subj_inputs = rnn.create_dataset(rnn.window, rnn.overlap, rnn.decimation)
		model, x_data_test, y_data_test, test_acc = rnn.train_session(save_model = False, earlystop = 0, train_split = 0.75, file_id = file_id)
		# create confusion matrix
		y_pred = np.argmax(model.predict(x_data_test), axis=-1)
		con_mat = tf.math.confusion_matrix(labels = y_data_test, predictions = y_pred).numpy()
		con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis = 1)[:, np.newaxis], decimals = 2)
		classes = ['N', 'AD']
		con_mat_df = pd.DataFrame(con_mat_norm, index = classes, columns = classes)
		figure = plt.figure(figsize = (5, 5))
		seaborn.heatmap(con_mat_df, annot = True, cmap = plt.cm.Blues)
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.tight_layout()
		plt.savefig('confusion_matrix.eps', format='eps')
