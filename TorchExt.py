import torch
from torch import nn
import toparray

class TorchExt:
	"""
	8bit textures should be reodered in TD first from RGBA to BGRA
	so that the arrays/torch tensor is in the correct order for the GPU.

	For networks sensitive to vertical direction the input/output should 
	be flipped vertically in TD.

	Careful with Python errors!!! Errors can cause illegal memory access
	with cudaMemory() and cause major memory leaks (recursive allocations)
		
	"""
	
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.to_tensorTop = self.ownerComp.op('to_tensor')
		self.device = torch.device('cuda')

		self.Setup()
		self.imageFilter = ImageFilter().to(self.device)

		
	def Setup(self):

		# create a stream for the device
		self.Stream = torch.cuda.Stream()

		# create a top array interface for the input
		self.input_array = toparray.TopArrayInterface(self.to_tensorTop, self.Stream.cuda_stream)

		cudamem = self.to_tensorTop.cudaMemory()

		# Using the .shape attribute of the cudamem object to get the shape of the array.
		# All these value can be set manually, see toparray.py for details on arguments
		self.Output = toparray.TopCUDAInterface(
			cudamem.shape.width,
			cudamem.shape.height,
			cudamem.shape.numComps,
			cudamem.shape.dataType, 
			self.Stream.cuda_stream)

		# set the output to the input on Setup
		with torch.cuda.stream(self.Stream):
			self.Output.image = torch.as_tensor(self.input_array, device=self.device)

	# called from execute DAT
	def ProcessTop(self):
		self.input_array.update(self.Stream.cuda_stream)

		with torch.cuda.stream(self.Stream):
			tensor = torch.as_tensor(self.input_array, device=self.device)
			with torch.no_grad():

				# uncomment to filter tensor. This only works with 32bit float data
				# tensor = self.imageFilter(tensor.unsqueeze(0)).squeeze(0)

				# optional normalize for filter output
				# tensor = self.imageFilter.normalize(tensor) 

				# copy tensor to output
				self.Output.image = tensor.permute(1, 2, 0).contiguous()


# Example filter for testing
class ImageFilter(nn.Module):
	"""
	Function to test io with TD TOPs
	"""
	def __init__(self, in_channels=4, out_channels=4, kernel_size=3):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
							  padding=kernel_size // 2, groups=4, bias=False)

		nn.init.constant_(self.conv.weight, 1. / (kernel_size ** 2) )
		# nn.init.normal_(self.conv.weight, 0.0, 1)
		# nn.init.xavier_uniform_(self.conv.weight, gain=2.0)

	def forward(self, x):
		# Assuming x is of shape [batch_size, channels, height, width]
		return self.conv(x)

	def normalize(self, tensor):
		tensor_min = tensor.min()
		tensor_max = tensor.max()
		normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
		return normalized_tensor



				
















