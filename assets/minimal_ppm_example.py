
import pdb
import numpy as np
import model.resnet as models
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
	def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
		""" """
		super(PPM, self).__init__()
		self.features = []
		for bin in bins:
			self.features.append(nn.Sequential(
				nn.AdaptiveAvgPool2d(bin),
				nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
				BatchNorm(reduction_dim),
				nn.ReLU(inplace=True)
			))
		self.features = nn.ModuleList(self.features)

	def forward(self, x):
		""" """
		x_size = x.size()
		out = [x]
		for f in self.features:
			out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
		return torch.cat(out, 1)




def ppm_main():
	"""   """
	fea_dim = 2048
	reduction_dim = int(fea_dim/len(bins))
	bins = (1, 2, 3, 6)
	BatchNorm = nn.BatchNorm2d

	ppm = PPM(in_dim=fea_dim, reduction_dim=reduction_dim, bins=bins, BatchNorm=BatchNorm)

	# input size
	x = 9999999999
	out = ppm(x)




class Backbone(nn.Module):
	def __init__(self, layers=50, 
						bins=(1, 2, 3, 6), 
						dropout=0.1, 
						classes=2, 
						zoom_factor=8, 
						use_ppm=True, 
						BatchNorm=nn.BatchNorm2d, 
						pretrained=False):
		""" """
		super(Backbone, self).__init__()
		assert layers in [50, 101, 152]
		assert 2048 % len(bins) == 0
		assert classes > 1
		assert zoom_factor in [1, 2, 4, 8]
		self.zoom_factor = zoom_factor
		self.use_ppm = use_ppm
		models.BatchNorm = BatchNorm

		if layers == 50:
			resnet = models.resnet50(pretrained=pretrained)
		elif layers == 101:
			resnet = models.resnet101(pretrained=pretrained)
		else:
			resnet = models.resnet152(pretrained=pretrained)
		self.layer0 = nn.Sequential(
			resnet.conv1, 
			resnet.bn1, 
			resnet.relu, 
			resnet.conv2, 
			resnet.bn2, 
			resnet.relu, 
			resnet.conv3, 
			resnet.bn3, 
			resnet.relu, 
			resnet.maxpool
		)
		self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

		for n, m in self.layer3.named_modules():
			if 'conv2' in n:
				m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)
		for n, m in self.layer4.named_modules():
			if 'conv2' in n:
				m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)

		fea_dim = 2048
		if use_ppm:
			self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
			fea_dim *= 2
		self.cls = nn.Sequential(
			nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
			BatchNorm(512),
			nn.ReLU(inplace=True),
			nn.Dropout2d(p=dropout),
			nn.Conv2d(512, classes, kernel_size=1)
		)
		if self.training:
			self.aux = nn.Sequential(
				nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
				BatchNorm(256),
				nn.ReLU(inplace=True),
				nn.Dropout2d(p=dropout),
				nn.Conv2d(256, classes, kernel_size=1)
			)

	def forward(self, x, y=None):
		"""
		"""
		x_size = x.size()
		assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
		h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
		w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x_tmp = self.layer3(x)
		x = self.layer4(x_tmp)

		return x





def backbone_test():
	"""
	"""
	train_h = 473
	train_w = 473
	pdb.set_trace()
	backbone = Backbone()
	# N, C, H, W
	input = np.ones((2,3,train_h,train_w), dtype=np.float32)
	input = torch.torch.from_numpy(input)
	output = backbone(input)





if __name__ == '__main__':
	backbone_test()
	#ppm_main()





