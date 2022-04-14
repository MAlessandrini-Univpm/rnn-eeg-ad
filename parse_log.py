import sys
import ast


fit_accuracy = []  # list of lists
fit_val_accuracy = []  # list of lists
test_accuracy = []  # list of numbers
time_train = []
time_test = []
oversample = False
pca = False
rpca = False
mspca = False
prefiltered = False
train_size = 0
pca_p = 0
spikes = 0


with open(sys.argv[1], 'r') as f:
	for line in f:
		if line.startswith('fit_accuracy '):
			fit_accuracy.append(ast.literal_eval(line[13:]))
		elif line.startswith('fit_val_accuracy '):
			fit_val_accuracy.append(ast.literal_eval(line[17:]))
		elif line.startswith('test_accuracy '):
			test_accuracy.append(float(line[13:]))
		elif line.startswith('time_train '):
			time_train.append(float(line[10:]))
		elif line.startswith('time_test '):
			time_test.append(float(line[9:]))
		elif line.startswith('window '):
			window = line[7:].strip()
		elif line.startswith('layers '):
			layers = line[7:].strip()
		elif line.startswith('oversample '):
			oversample = ast.literal_eval(line[11:])
		elif line.startswith('pca '):
			pca = ast.literal_eval(line[4:])
		elif line.startswith('rpca '):
			rpca = ast.literal_eval(line[5:])
		elif line.startswith('mspca '):
			mspca = ast.literal_eval(line[6:])
		elif line.startswith('prefiltered '):
			prefiltered = ast.literal_eval(line[12:])
		elif line.startswith('train_size '):
			train_size = ast.literal_eval(line[11:])
		elif line.startswith('p '):
			pca_p = ast.literal_eval(line[2:])
		elif line.startswith('spikes '):
			spikes = ast.literal_eval(line[7:])

n = len(fit_accuracy)
assert((n == len(fit_val_accuracy) or not fit_val_accuracy) and (n == len(test_accuracy) or not test_accuracy) and n == len(time_train) and (n ==len(time_test) or not time_test))

if not n: sys.exit(0)  # log empty if training interrupted

a_trn = [0., 0., 0.]  # average of average, average of max, average of last
a_val = [0., 0., 0.]  # average of average, average of max, average of last
a_tst = [0., 0.]  # average, max

a_trn[0] = sum([sum(l) / len(l) for l in fit_accuracy]) / n
a_trn[1] = sum([max(l) for l in fit_accuracy]) / n
a_trn[2] = sum([l[-1] for l in fit_accuracy]) / n
a_val[0] = sum([sum(l) / len(l) for l in fit_val_accuracy]) / n
a_val[1] = sum([max(l) for l in fit_val_accuracy]) / n
a_val[2] = sum([l[-1] for l in fit_val_accuracy]) / n
a_tst[0] = sum(test_accuracy) / n
a_tst[1] = max(test_accuracy) if test_accuracy else 0

print(sys.argv[1], layers, window, 'OVR' if oversample else '', 'PCA' if pca else '', 'RPCA' if rpca else '', 'MSPCA' if mspca else '', 'PRE' if prefiltered else '')
print(f'N {train_size}')
print(f'p {pca_p}')
print(f'spikes {spikes}')
print(f'[{n:2d}]     avg      max     last')
print(f'train {a_trn[0]:7.4f}  {a_trn[1]:7.4f}  {a_trn[2]:7.4f}')
print(f'val   {a_val[0]:7.4f}  {a_val[1]:7.4f}  {a_val[2]:7.4f}')
print(f'test  {a_tst[0]:7.4f}  {a_tst[1]:7.4f}')
print(f'trn_time {sum(time_train) / n:.1f}')
print(f'tst_time {sum(time_test) / n:.1f}')
