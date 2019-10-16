

import matplotlib.pyplot as plt
import numpy as np
import pdb




def compute_autocorrelation(I, patch, window_sz=3):
	""" """
	img_h, img_w = I.shape
	window_h, window_w = window_sz, window_sz

	shift_range_x = img_w - window_w + 1
	shift_range_y = img_h - window_h + 1
	error_grid = np.zeros((shift_range_y, shift_range_x))

	#pdb.set_trace()

	for u in range(shift_range_x):
		for v in range(shift_range_y):
			# patch shift error
			error_grid[v,u] = np.square(I[v:v+window_h,u:u+window_w] - patch).sum()

	print(error_grid)
	plt.imshow(error_grid)
	plt.gray()
	plt.show()



def main():
	"""
	Window should have odd size, like filter, to have anchor point.
	"""
	# I = np.array(
	# 	[
	# 		[0,0,0,0,0,0],
	# 		[0,0,0,0,0,0],
	# 		[0,0,0,0,0,0],
	# 		[0,0,0,0,0,0],
	# 		[0,0,1,1,1,1],
	# 		[0,0,1,1,1,1],
	# 		[0,0,1,1,1,1],
	# 		[0,0,1,1,1,1],
	# 		[0,0,0,0,0,0]
	# 	])

	
	#patch = I[-4:-1,1:4] # very bottom, along edge
	#patch = I[-6:-3,1:4] # middle, on corner
	#patch = I[1:4,1:4] # top, flat region
	


	# I = np.array(
	# 	[
	# 		[0,255,255,255,255,255,0],
	# 		[0,  0,  0,  0,  0,  0,0],
	# 		[0,255,255,255,255,255,0],
	# 		[0,  0,  0,  0,  0,  0,0],
	# 		[0,255,255,255,255,255,0],
	# 		[0,  0,  0,  0,  0,  0,0],
	# 		[0,255,255,255,255,255,0],
	# 		[0,  0,  0,  0,  0,  0,0],
	# 		[0,255,255,255,255,255,0],
	# 		[0,  0,  0,  0,  0,  0,0],
	# 		[0,255,255,255,255,255,0],
	# 		[0,  0,  0,  0,  0,  0,0],
	# 		[0,255,255,255,255,255,0],
	# 		[0,  0,  0,  0,  0,  0,0],
	# 		[0,255,255,255,255,255,0],
	# 		[0,  0,  0,  0,  0,  0,0]
	# 	])

	I = np.array(
		[
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,  0,  0,  0,  0,  0,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0],
			[0,255,255,255,255,255,0]
		])


	patch = I[2:7,1:6]

	plt.imshow(I)
	plt.gray()
	plt.show()
	compute_autocorrelation(I, patch, window_sz=5)
	#compute_autocorrelation(I, patch, window_sz=3)

	#grad_mag = plot_imgrad(I, approx_method='np_diff')
	grad_mag = plot_imgrad(I, approx_method='np_grad')
	plt.imshow(grad_mag)
	plt.gray()
	plt.show()


def plot_imgrad(I, approx_method='np_grad'):
	""" """
	if approx_method == 'np_grad':
		dy,dx = np.gradient(I)

	elif approx_method == 'np_diff':
		dx = np.zeros(I.shape)
		dy = np.zeros(I.shape)
		dx[:, 1:] = np.diff(I, n=1, axis=1) # compute gradient on x-direction
		dy[1:, :] = np.diff(I, n=1, axis=0) # compute gradient on y-direction

	# print('dy')
	# print(dy)
	# print('dx')
	# print(dx)
	# plt.subplot(1,2,1)
	# plt.imshow(dx)
	# plt.subplot(1,2,2)
	# plt.imshow(dy)
	# plt.show()

	# now try one with change in gradient not equal to high error
	return np.sqrt( np.square(dx) + np.square(dy) )

if __name__ == '__main__':
	main()
