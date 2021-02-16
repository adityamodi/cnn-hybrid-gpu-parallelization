import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = 'training_curve_plot_num_workers_2.png'
FILE_NAME_A_1 = 'results_mnist/scheme_a/num_workers_2/batch_size_32/training_loss.txt'
FILE_NAME_B_1 = 'results_mnist/scheme_b/num_workers_2/batch_size_32/training_loss.txt'
FILE_NAME_C_1 = 'results_mnist/scheme_c/num_workers_2/batch_size_32/training_loss.txt'

FILE_NAME_A_2 = 'results_mnist/scheme_a/num_workers_3/batch_size_32/training_loss.txt'
FILE_NAME_B_2 = 'results_mnist/scheme_b/num_workers_3/batch_size_32/training_loss.txt'
FILE_NAME_C_2 = 'results_mnist/scheme_c/num_workers_3/batch_size_32/training_loss.txt'

FILE_NAME_A_3 = 'results_mnist/scheme_a/num_workers_4/batch_size_32/training_loss.txt'
FILE_NAME_B_3 = 'results_mnist/scheme_b/num_workers_4/batch_size_32/training_loss.txt'
FILE_NAME_C_3 = 'results_mnist/scheme_c/num_workers_4/batch_size_32/training_loss.txt'

X_AXIS_LABEL = '# Iterations'
Y_AXIS_LABEL = 'Cross-Entropy Training Loss'
ITERATIONS = 10000
TEST_INTERVAL = 100

SCHEMES = ['# workers = 2(A)', '# workers = 2(B)', '# workers = 2(C)',
			'# workers = 3(A)', '# workers = 3(B)', '# workers = 3(C)',
			'# workers = 4(A)', '# workers = 4(B)', '# workers = 4(C)',]

def main():
	x = np.arange(ITERATIONS / TEST_INTERVAL)
	y = np.loadtxt(FILE_NAME_A_1)
	y = y * 1. / 32.
	plt.plot(x, y, label=SCHEMES[0])

	y = np.loadtxt(FILE_NAME_B_1)
	plt.plot(x, y, label=SCHEMES[1])

	y = np.loadtxt(FILE_NAME_C_1)
	plt.plot(x, y, label=SCHEMES[2])

	y = np.loadtxt(FILE_NAME_A_2)
	y = y * 1. / 32.
	plt.plot(x, y, label=SCHEMES[3])

	y = np.loadtxt(FILE_NAME_B_2)
	plt.plot(x, y, label=SCHEMES[4])

	y = np.loadtxt(FILE_NAME_C_2)
	plt.plot(x, y, label=SCHEMES[5])

	y = np.loadtxt(FILE_NAME_A_3)
	y = y * 1. / 32.
	plt.plot(x, y, label=SCHEMES[6])

	y = np.loadtxt(FILE_NAME_B_3)
	plt.plot(x, y, label=SCHEMES[7])

	y = np.loadtxt(FILE_NAME_C_3)
	plt.plot(x, y, label=SCHEMES[8])

	plt.xlabel(X_AXIS_LABEL)
	plt.ylabel(Y_AXIS_LABEL)
	plt.legend(loc='best')

	plt.savefig(FILE_NAME)

if __name__ == '__main__':
	main()