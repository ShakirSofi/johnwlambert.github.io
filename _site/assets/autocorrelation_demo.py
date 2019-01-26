
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

def normalize_to_uint8(e_ssd):
	e_ssd -= np.amin(e_ssd)
	e_ssd /= np.amax(e_ssd)
	e_ssd *= 255
	e_ssd = e_ssd.astype(np.uint8)
	return e_ssd

def compute_e_ssd(img_h, img_w, window_h, window_w, mtn_img, patch_img):
	""" """
	e_ssd = np.zeros((img_h-window_h,img_w-window_w))
	start = time.time()
	# start at top-left corner
	for u in range(0,img_h-window_h,5):
		for v in range(0,img_w-window_w,5):
			e_ssd[u,v] = np.square(mtn_img[u:u+window_h:,v:v+window_w,:] - patch_img).sum()

	end = time.time()
	print('Took: ', end-start)

	e_ssd = normalize_to_uint8(e_ssd)
	np.save('e_ssd_centered', e_ssd)

def plot_2d_matrix(mat):
	""" """
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	import matplotlib.pyplot as plt
	from matplotlib.ticker import LinearLocator, FormatStrFormatter

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	Y,X = np.meshgrid( np.arange(0,mat.shape[0],1), np.arange(0,mat.shape[1],1))
	Z = mat[Y,X]
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
		linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()


def ssd_main():
	base_dir = '/Users/jlambert-admin/Desktop'

	patch_fpath = 'mtn_img_centered_patch.png'
	mtn_img_fpath = 'mtn_img_centered_full.png'

	patch_img = imageio.imread(f'{base_dir}/{patch_fpath}')[:,:,:3] # throw away 4th channel
	mtn_img = imageio.imread(f'{base_dir}/{mtn_img_fpath}')[:,:,:3] # throw away 4th channel

	window_h = patch_img.shape[0]
	window_w = patch_img.shape[1]

	img_h = mtn_img.shape[0]
	img_w = mtn_img.shape[1]

	#compute_e_ssd(img_h, img_w, window_h, window_w, mtn_img, patch_img)
	e_ssd = np.load('e_ssd_centered.npy')

	for row in range(e_ssd.shape[0]):
		for col in range(e_ssd.shape[1]):
			if (row%5 != 0) or (col%5 != 0):
				e_ssd[row,col] = e_ssd[int(5 * np.floor(float(row)/5)), int(5 * np.floor(float(col)/5))]

	plot_2d_matrix(e_ssd)

	fig = plt.figure()
	ax = fig.gca()
	ax.imshow(mtn_img)
	row,col = np.vstack(np.argwhere(e_ssd == np.min(e_ssd))).mean(axis=0).astype(np.int32)

	# bounding box points
	bbox = np.array([col,row,col+window_w,row+window_h])
	xmin, ymin, xmax, ymax = bbox.astype(np.int32).squeeze()
	bbox_h, bbox_w, _ = mtn_img[ymin:ymax,xmin:xmax].shape

	color = np.array([255,0,0])
	tiled_color = np.tile(color.reshape(1,1,3),(bbox_h,bbox_w,1))
	mtn_img[ymin:ymax,xmin:xmax,:] = (mtn_img[ymin:ymax,xmin:xmax,:] + tiled_color)/2.

	plt.imshow(mtn_img)
	plt.show()

def rgb2gray(rgb):
	"""Convert RGB image to grayscale
	Args:
	- rgb: A numpy array of shape (m,n,c) representing an RGB image
	Returns:
	- gray: A numpy array of shape (m,n) representing the corresponding grayscale image
	"""
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def gaussian_kernel(sz,sigma=1.):
    """
    Create Gaussian kernel with side length sz and standard
    deviation of sigma.
    """
    ax = np.arange(-sz // 2 + 1., sz // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel # / np.sum(kernel)


def im_gradients_main():
	""" """
	base_dir = '/Users/jlambert-admin/Desktop'
	mtn_img_fpath = 'tiger.jpg' # 'mtn_img_centered_full.png'
	mtn_img = imageio.imread(f'{base_dir}/{mtn_img_fpath}') #[:,:,:3] # throw away 4th channel

	mtn_img = rgb2gray(mtn_img)
	mtn_img = mtn_img.astype(np.uint8)

	img_h = mtn_img.shape[0]
	img_w = mtn_img.shape[1]

	import cv2
	ksize = 2
	sigma = 4

	g = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
	small_filter = g * g.T
	#small_filter = gaussian_kernel(ksize,sigma)

	#plot_2d_matrix(small_filter)

	kx = np.zeros(mtn_img.shape)
	ky = np.zeros(mtn_img.shape)
	kx[:, :-1] = np.diff(mtn_img, n=1, axis=1) # compute gradient on x-direction
	ky[:-1, :] = np.diff(mtn_img, n=1, axis=0) # compute gradient on y-direction
	# import cv2
	# kx = cv2.Sobel(mtn_img, cv2.CV_64F, 1, 0, ksize=3) / 8.
	# ky = cv2.Sobel(mtn_img, cv2.CV_64F, 0, 1, ksize=3) / 8.

	# http://vision.stanford.edu/teaching/cs131_fall1617/lectures/lecture5_edges_cs131_2016.pdf
	# import scipy.signal
	# kx = scipy.signal.convolve2d(  in1 = kx,
 #                                        in2 = small_filter,
 #                                        mode = 'same',
 #                                        boundary = 'fill',
 #                                        fillvalue = 0) / 8.

	# ky = scipy.signal.convolve2d(  in1 = ky,
 #                                        in2 = small_filter,
 #                                        mode = 'same',
 #                                        boundary = 'fill',
 #                                        fillvalue = 0) / 8.

	gradient_mag = np.square(kx) + np.square(ky)
	gradient_mag = normalize_to_uint8(gradient_mag)
	# plt.hist(gradient_mag.flatten(),bins=100)
	# plt.show()

	plt.gray()
	plt.subplot(1,2,1)
	plt.imshow(mtn_img)

	plt.subplot(1,2,2)
	plt.imshow(ky)#gradient_mag)
	plt.show()





#ssd_main()
im_gradients_main()
