# Animated Sparse Voxel Octrees @ CUDA


## Introduction

Animated Sparse Voxel Octrees (ASVO) is an animation technique for sparse voxel octrees. It allows you to apply rigid transformations or skinning to voxel based 3D models.

You can find more detailed information on the [project page](http://bautembach.de/wordpress/?page_id=7).

This is the high performance CUDA implementation (aka ASVO@CUDA) of the C# prototype (aka ASVO).


## Dependencies

- Windows 7 **64bit**
- [CUDA 5](https://developer.nvidia.com/cuda-downloads)
- CUDA capable GPU with Compute Capability 3.0 or higher, 1GB VRAM or more.
- Visual Studio 2010


## Usage

After launching the program you can rotate the camera by holding down the *left* mouse button and moving your mouse. Holding down the *right* mouse button and moving your mouse allows you to zoom in and out.


## Known Issues

- Unnecessary Windows dependency (just for QueryHighPerformanceTimer and freeglut).
- Broken 32bit build.
- Outdated makefile 'standard.make'
- **NOT(!) platform-independant: Code assumes that 'long' refers to 32bit integers**.


## Roadmap

1. Remove Windows dependencies and make code platfrom-independant (remove assumptions about size and layout/endianness of types, provide make-file, etc.).
2. Improve/Extend algorithm.