# TopArray

TopArray is a Python module designed to facilitate the interaction between Python-based tensor operations and TouchDesigner (TD) TOPs (Texture Operators). It provides a workflow for referencing CUDA memory from TOPs as tensors and for copying tensor data back to a script TOP.

## Installation

To use TopArray, ensure you have TD compatible Python and PyTorch with CUDA installed. The Python installation/environment needs to be visible to TouchDesigner.

## How To Use

Load ExampleTensorIO.toe and toggle the Active parameter on the TensorIO componenent. See component network and TorchExt.py for details on how to interact with the TopArrayInterface and the TopCUDAInterface.

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

- **top** (`td.TOP`): The TOP to copy the data to.
- **stream** (`int`, optional): The CUDA stream to use for transfers. Defaults to `0`.

### Methods

#### `copyFromTensor(self, scriptOP)`

Copies data from a tensor to the specified ScriptTOP.

- **Parameters**:
  - `scriptOP` (`td.scriptOP`): The ScriptTOP to copy the tensor to.

Wraps the `copyCUDAMemory` method to copy the tensor data into the ScriptTOP's memory.



