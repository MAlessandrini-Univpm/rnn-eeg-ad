import rnn_eeg_ad as rnn
import math

# Note: we create global variables in the rnn module, because functions there expect that

rnn.dense1 = 0
rnn.lstm1 = 8
rnn.lstm2 = 8
rnn.lstm3 = 0

rnn.subjs_train_perm = ( (tuple(i for i in range(2, 15)) + tuple(i for i in range(18, 35)), ()), )
rnn.subjs_test = (0, 1, 15, 16, 17)
rnn.decimation = 0
rnn.epochs = 20
rnn.oversample = True
rnn.window = 256
rnn.overlap = rnn.window // 2
rnn.pca = True
rnn.rpca = True


def f(mu):
	rnn.rpca_mu = mu
	rnn.x_data, rnn.y_data, rnn.subj_inputs = rnn.create_dataset(rnn.window, rnn.overlap, rnn.decimation)
	model, x_data_test, y_data_test, test_acc = rnn.train_session(save_model = False, write_report = False)
	return -test_acc  # GSS finds minimum


def gss(f):
	a = 0.01
	b = 1

	# GSS implementation
	
	invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
	invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
	h = b - a
	c = a + invphi2 * h
	d = a + invphi * h
	yc = f(c)
	nn = rnn.norm_s
	yd = f(d)
	results = [(c,yc,nn),(d,yd,rnn.norm_s)]
	print(results)
	
	for k in range(8):  # max number of iterations
		if yc < yd:
			b = d
			d = c
			yd = yc
			h = invphi * h
			c = a + invphi2 * h
			yc = f(c)
			results.append((c, yc, rnn.norm_s))
			print(results)
		else:
			a = c
			c = d
			yc = yd
			h = invphi * h
			d = a + invphi * h
			yd = f(d)
			results.append((d, yd, rnn.norm_s))
			print(results)
	
	f_out = open('mu_results.txt', 'a')
	print('spikes:', rnn.spikes, file = f_out)
	for r in results: print(r, file = f_out)
	print('range:', file = f_out)
	if yc < yd:
		print((a, d), file = f_out)
	else:
		print((c, b), file = f_out)
	f_out.close()


for rnn.spikes in (1/1000, 1/500, 1/200, 1/100):
	gss(f)

