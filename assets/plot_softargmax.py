


import numpy as np
import pdb
import seaborn as sns
import matplotlib.pylab as plt


def render_heatmap(grid):
	""" """
	if grid.ndim == 1:
		grid = grid.reshape(1,-1)

	ax = sns.heatmap(grid,  linewidth=0.5) # yticklabels=ylabels, xticklabels=xlabels,
	# plt.xlabel('Max. Allowed Track Age Since Update')
	# plt.ylabel('Minimum Number of Required Hits for Track Birth')
	plt.axis('equal')
	plt.show()
	#return ax




def soft_argmax_1d():
	""" """
	y = np.arange(4)
	S = np.array([0.1,0.7, 0.05, 0.15])

	y = np.arange(100)
	S = np.random.rand(10,10)

	y = np.arange(25) #25)
	S = np.array(
		[
			[0.1, 0.2, 0.3, 0.4, 0.2],
			[0.1, 0.2, 0.3, 0.4, 0.3],
			[0.1, 0.2, 0.3, 0.5, 0.3],
			[0.1, 0.2, 0.3, 0.3, 0.3],
			[0.1, 0.2, 0.2, 0.2, 0.2]
		])

	render_heatmap(S)
	S = S.flatten()

	for temp in [10,100]:

		num = np.sum( np.exp(S * temp) * y)
		denom = np.sum( np.exp(S * temp))

		mass = np.exp(S * temp)
		mass = mass.reshape(5,5) / denom
		print(np.round(mass,2))
		# pdb.set_trace()
		render_heatmap(mass)
		
		x = num / denom
		row = x // 5
		col = x % 5
		print(row,col)

def soft_argmax_2d():
	""" """
	S = np.array(
		[
			[0.1, 0.2, 0.3, 0.4, 0.2],
			[0.1, 0.2, 0.3, 0.4, 0.3],
			[0.1, 0.2, 0.3, 0.5, 0.3],
			[0.1, 0.2, 0.3, 0.3, 0.3],
			[0.1, 0.2, 0.2, 0.2, 0.2]
		])

	# S = np.array(
	# 	[
	# 		[0.1, 0.2, 0.2, 0.2, 0.2],
	# 		[0.1, 0.2, 0.3, 0.3, 0.3],
	# 		[0.1, 0.2, 0.3, 0.9, 0.3],
	# 		[0.1, 0.2, 0.3, 0.3, 0.3],
	# 		[0.1, 0.2, 0.2, 0.2, 0.2]
	# 	])

	# S = np.array(
	# 	[
	# 		[0, 0, 0, 0.9, 0.0],
	# 		[0, 0, 0, 0.8, 0.0],
	# 		[0, 0, 0, 0.7, 0.0],
	# 		[0, 0, 0, 0.6, 0.0],
	# 		[0, 0, 0, 0.5, 0.0]
	# 	])

	render_heatmap(S)
	N = 5
	for temp in [1,10,100]:
		denom = np.sum( np.exp(S * temp))
		mass = np.exp(S * temp)
		mass = mass.reshape(N,N) / denom
		m = mass

		W = np.arange(N)
		W = np.tile(W,(N,1))
		

		x = (W*m).sum()
		y = (W.T*m).sum()
		print(y,x)



if __name__ == '__main__':
	#soft_argmax_1d()
	soft_argmax_2d()