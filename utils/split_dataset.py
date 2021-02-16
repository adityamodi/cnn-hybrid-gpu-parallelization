import os
import struct
import random
import numpy as np
import cPickle as pickle

TRAINING_DATASET_PATH = 'train-images-idx3-ubyte'
TRAINING_LABEL_PATH = 'train-labels-idx1-ubyte'
TESTING_DATASET_PATH = 't10k-images-idx3-ubyte'
TESTING_LABEL_PATH = 't10k-labels-idx1-ubyte'
NUM_PARTITIONS = 4

CIFAR_PATH = 'cifar-10-batches-py/{}'

def readDataset(dataset_path=None, label_path=None):
	print("dataset and labels to process: {} and {}".format(dataset_path, label_path))
	with open(label_path, 'rb') as flbl:
		magic, num = struct.unpack(">II", flbl.read(8))
		lbl = np.fromfile(flbl, dtype=np.int8)
	with open(dataset_path, 'rb') as fimg:
		magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

	get_img = lambda idx: (lbl[idx], img[idx])

	for i in xrange(len(lbl)):
		yield get_img(i)

def mainMNIST():
	data = list(readDataset(dataset_path=TRAINING_DATASET_PATH, label_path=TRAINING_LABEL_PATH))
	dataset_size = len(data)
	partition_size = int(dataset_size / NUM_PARTITIONS)
	x = np.arange(dataset_size)
	random.shuffle(x)
	for i in range(NUM_PARTITIONS):
		# with open('mnist_training_images_{}.csv'.format(i)) as fimgs:
		fimgs = open('mnist_training_images_{}.csv'.format(i), 'wb')
		fimgs.write('{} {}\n'.format(partition_size, 28 * 28))
		flabels = open('mnist_training_labels_{}.csv'.format(i), 'wb')
		flabels.write('{}\n'.format(partition_size))

		partition_idxes = x[i * partition_size : (i + 1) * partition_size]
		img_arr = np.zeros((partition_size, 28*28))
		label_arr = np.zeros((partition_size))
		count = 0
		for j in partition_idxes:
			lab, img = data[j]
			img_arr[count, :] = img.flatten().astype(int)
			label_arr[count] = lab.astype(int)
			count += 1
		# np.savetxt('mnist_training_images_{}.csv'.format(i), img_arr, delimiter=' ')
		# np.savetxt('mnist_training_labels_{}.csv'.format(i), label_arr, delimiter=' ')
		np.savetxt(fimgs, img_arr, fmt='%d', delimiter=' ')
		np.savetxt(flabels, label_arr, fmt='%d', delimiter=' ')
		fimgs.close()
		flabels.close()

	data = list(readDataset(dataset_path=TESTING_DATASET_PATH, label_path=TESTING_LABEL_PATH))
	dataset_size = len(data)
	# partition_size = int(dataset_size / NUM_PARTITIONS)
	x = np.arange(dataset_size)
	fimgs = open('mnist_training_images_{}.csv'.format(i), 'wb')
	fimgs.write('{} {}\n'.format(partition_size, 28 * 28))
	flabels = open('mnist_training_labels_{}.csv'.format(i), 'wb')
	flabels.write('{}\n'.format(partition_size))
	random.shuffle(x)

	fimgs = open('mnist_testing_images.csv', 'wb')
	fimgs.write('{} {}\n'.format(partition_size, 28 * 28))
	flabels = open('mnist_testing_labels.csv', 'wb')
	flabels.write('{}\n'.format(partition_size))

	partition_idxes = x[0 : ]
	img_arr = np.zeros((partition_size, 28*28))
	label_arr = np.zeros((partition_size))
	count = 0
	for j in partition_idxes:
		lab, img = data[j]
		img_arr[count, :] = img.flatten().astype(int)
		label_arr[count] = lab.astype(int)
		count += 1
	np.savetxt(fimgs, img_arr, fmt='%d', delimiter=' ')
	np.savetxt(flabels, label_arr, fmt='%d', delimiter=' ')
	fimgs.close()
	flabels.close()

def mainCIFAR10():
	img_list = []
	label_list = []
	with open(CIFAR_PATH.format('data_batch_1'), 'rb') as fd:
		dataset_dict = pickle.load(fd)
	img_list.extend(dataset_dict['data'])
	label_list.extend(dataset_dict['labels'])
	
	with open(CIFAR_PATH.format('data_batch_2'), 'rb') as fd:
		dataset_dict = pickle.load(fd)
	img_list.extend(dataset_dict['data'])
	label_list.extend(dataset_dict['labels'])

	with open(CIFAR_PATH.format('data_batch_3'), 'rb') as fd:
		dataset_dict = pickle.load(fd)
	img_list.extend(dataset_dict['data'])
	label_list.extend(dataset_dict['labels'])

	with open(CIFAR_PATH.format('data_batch_4'), 'rb') as fd:
		dataset_dict = pickle.load(fd)
	img_list.extend(dataset_dict['data'])
	label_list.extend(dataset_dict['labels'])

	with open(CIFAR_PATH.format('data_batch_5'), 'rb') as fd:
		dataset_dict = pickle.load(fd)
	img_list.extend(dataset_dict['data'])
	label_list.extend(dataset_dict['labels'])

	img_arr = np.array(img_list)
	label_arr = np.array(label_list)

	print('image array.shape: {}'.format(img_arr.shape))
	print('label array.shape: {}'.format(label_arr.shape))

	dataset_size = img_arr.shape[0]
	partition_size = int(dataset_size / NUM_PARTITIONS)
	x = np.arange(dataset_size)
	random.shuffle(x)
	for i in range(NUM_PARTITIONS):
		# with open('mnist_training_images_{}.csv'.format(i)) as fimgs:
		fimgs = open('cifar_training_images_{}.csv'.format(i), 'wb')
		fimgs.write('{} {}\n'.format(partition_size, 32 * 32 * 3))
		flabels = open('cifar_training_labels_{}.csv'.format(i), 'wb')
		flabels.write('{}\n'.format(partition_size))

		partition_idxes = x[i * partition_size : (i + 1) * partition_size]
		img_tostore = np.zeros((partition_size, 32*32*3))
		label_tostore = np.zeros((partition_size))
		count = 0
		for j in partition_idxes:
			img = img_arr[j, :]
			lab = label_arr[j]
			# lab, img = data[j]
			img_tostore[count, :] = img.flatten().astype(int)
			label_tostore[count] = lab.astype(int)
			count += 1
		# np.savetxt('mnist_training_images_{}.csv'.format(i), img_arr, delimiter=' ')
		# np.savetxt('mnist_training_labels_{}.csv'.format(i), label_arr, delimiter=' ')
		np.savetxt(fimgs, img_tostore, fmt='%d', delimiter=' ')
		np.savetxt(flabels, label_tostore, fmt='%d', delimiter=' ')
		fimgs.close()
		flabels.close()

	img_list = []
	label_list = []
	with open(CIFAR_PATH.format('test_batch'), 'rb') as fd:
		dataset_dict = pickle.load(fd)
	img_list.extend(dataset_dict['data'])
	label_list.extend(dataset_dict['labels'])
	
	img_arr = np.array(img_list)
	label_arr = np.array(label_list)

	dataset_size = img_arr.shape[0]
	# partition_size = int(dataset_size / NUM_PARTITIONS)
	x = np.arange(dataset_size)

	fimgs = open('cifar_testing_images.csv', 'wb')
	fimgs.write('{} {}\n'.format(partition_size, 32 * 32 * 3))
	flabels = open('cifar_testing_labels.csv', 'wb')
	flabels.write('{}\n'.format(partition_size))

	random.shuffle(x)

	partition_idxes = x[0 : ]
	img_tostore = np.zeros((partition_size, 32*32*3))
	label_tostore = np.zeros((partition_size))
	count = 0
	for j in partition_idxes:
		img = img_arr[j, :]
		lab = label_arr[j]
		img_tostore[count, :] = img.flatten().astype(int)
		label_tostore[count] = lab.astype(int)
		count += 1
	np.savetxt(fimgs, img_tostore, fmt='%d', delimiter=' ')
	np.savetxt(flabels, label_tostore, fmt='%d', delimiter=' ')
	fimgs.close()
	flabels.close()

if __name__ == '__main__':
	# mainMNIST()
	mainCIFAR10()
