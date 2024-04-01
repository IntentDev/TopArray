'''
MIT License

Copyright (c) 2024 Keith Lostracco

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import torch

def np_dtype_to_torch(np_dtype):
    type_map = {
        np.float64: torch.float64,
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int64: torch.int64,
        np.int32: torch.int32,
        np.int16: torch.int16,	
        np.int8: torch.int8,
        np.uint8: torch.uint8
    }
    return type_map[np_dtype]

def create_dtype_descr_map():
	dtype_descr_map = {}
	for category, types in np.sctypes.items():
		if category in ['int', 'uint', 'float', 'complex']:
			for t in types:
				dtype_instance = np.dtype(t)
				dtype_descr_map[t] = {'descr': dtype_instance.descr, 'num_bytes': dtype_instance.itemsize}
	return dtype_descr_map

NP_TYPE_MAP = create_dtype_descr_map()

class TopArrayInterface:
	"""
	wrapper for a TD TOP that implements the __cuda_array_interface__ protocol

	Parameters
	----------
	top : td.TOP
		The TOP to wrap
	stream : int, optional
		The CUDA stream to use for transfers. The default is 0.

	"""

	def __init__(self, top, stream=0):
		self.top = top
		mem = top.cudaMemory(stream=stream)
		self.w, self.h = mem.shape.width, mem.shape.height
		self.num_comps = mem.shape.numComps
		self.dtype = mem.shape.dataType
		shape = (mem.shape.numComps, self.h, self.w)
		dtype_info = NP_TYPE_MAP[mem.shape.dataType]
		dtype_descr = dtype_info['descr']
		num_bytes = dtype_info['num_bytes']
		num_bytes_px = num_bytes * mem.shape.numComps
		
		self.__cuda_array_interface__ = {
			"version": 3,
			"shape": shape,
			"typestr": dtype_descr[0][1],
			"descr": dtype_descr,
			"stream": stream,
			"strides": (num_bytes, num_bytes_px * self.w, num_bytes_px),
			"data": (mem.ptr, False),
		}

	def update(self, stream=0):
		"""
		Updates the __cuda_array_interface__ with updated memory from the TOP

		New 

		Parameters
		----------
		top : td.TOP
			The TOP to wrap
		stream : int, optional
			The CUDA stream to use for transfers. The default is 0.

		"""
		mem = self.top.cudaMemory(stream=stream)
		self.__cuda_array_interface__['stream'] = stream
		self.__cuda_array_interface__['data'] = (mem.ptr, False)
		return
	
class TopCUDAInterface:
	"""
	Handles copying data from an array or tensor to a TD TOP

	Parameters
	----------
	width : int
		The width of the array

	height : int
		The height of the array

	num_comps : int
		The number of components in the array
		
	dtype : np.dtype
		The data type of the array

	stream : int, optional
		The CUDA stream to use for transfers. The default is 0.

	"""

	def __init__(self, width, height, num_comps, dtype, stream=0):
		self.image = None
		self.stream = stream
		self.mem_shape = CUDAMemoryShape()
		self.mem_shape.width = width
		self.mem_shape.height = height
		self.mem_shape.numComps = num_comps
		self.mem_shape.dataType = dtype
		self.bytes_per_comp = np.dtype(dtype).itemsize
		self.size = width * height * num_comps * self.bytes_per_comp

	def copyFromTensor(self, scriptOP):
		"""
		Wrapper for copyCUDAMemory that copies a data to a TD TOP

		Parameters
		----------
		scriptOP : td.scriptOP
			The scriptTOP to copy the tensor to
		"""
		if self.image is not None:
			scriptOP.copyCUDAMemory(
				self.image.data_ptr(), 
				self.size, 
				self.mem_shape,
				stream=self.stream
			)
		return