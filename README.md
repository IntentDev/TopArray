# TopArray

TopArray is a Python module designed to facilitate the interaction between Python/PyTorch tensor operations and TouchDesigner TOPs. It provides a workflow for referencing CUDA memory from TOPs as tensors and for copying a tensor's data to a scriptTOP efficiently.

## Installation

To use TopArray, ensure you have TD compatible Python and PyTorch with CUDA installed. The Python installation/environment needs to be visible to TouchDesigner.

## How To Use

Use `git clone https://github.com/IntentDev/TopArray.git` or download zip file of the repository. 

Load ExampleTensorIO.toe and toggle the Active parameter on the TensorIO componenent. 

See component network and TorchExt.py for details on how to interact with the TopArrayInterface and the TopCUDAInterface.

# Documentation

## TopArrayInterface

A wrapper class for a TouchDesigner TOP that implements the `__cuda_array_interface__` protocol, allowing direct integration with CUDA-aware libraries.

### Parameters

- **top** (`td.TOP`): The TOP to wrap.
- **stream** (`int`, optional): The CUDA stream to use for transfers. Defaults to `0`.


#### `update(self, top, stream=0)`

Updates the `__cuda_array_interface__` with memory from the TOP.

- **Parameters**:
  - `stream`: The CUDA stream for transfers. Defaults to `0`.

Updates the `stream` and `data` keys in the `__cuda_array_interface__`.

## TopCUDAInterface

Handles copying data from an array or tensor to a TouchDesigner TOP.

### Parameters

- **width** (`int`): width of the array that will be used to set the image
- **height** (`int`): height of the array that will be used to set the image
- **num_comps** (`int`): number of color components/channels in the array
- **dtype** (`np.dtype`): NumPy data type of the array
- **stream** (`int`, optional): The CUDA stream to use for transfers. Defaults to `0`.

### Methods

#### `copyFromTensor(self, scriptOP)`

Copies data from a tensor to the specified ScriptTOP.

- **Parameters**:
  - `scriptOP` (`td.scriptOP`): The ScriptTOP to copy the tensor to.

Wraps the `copyCUDAMemory` method to copy the tensor data into the ScriptTOP's memory.



