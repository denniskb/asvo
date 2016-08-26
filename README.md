# Animated Sparse Voxel Octrees @ CUDA


## Introduction

Animated Sparse Voxel Octrees (ASVO) is an animation technique for sparse voxel octrees. It allows you to apply rigid transformations or skinning to voxel based 3D models.

You can find more detailed information on the [project page](http://bautembach.de/wordpress/?page_id=7).

This is the high performance CUDA implementation (aka ASVO@CUDA) of the C# prototype (aka ASVO).


## Dependencies

- CMAKE 3.0.0
- CUDA 7.5
- CUDA capable GPU with Compute Capability 3.0 or higher, 1GB VRAM or more.
- freeglut 3.0.0 or later
- GLEW 1.12.0 or later


## Usage

After launching the program you can rotate the camera by holding down the *left* mouse button and moving your mouse. Holding down the *right* mouse button and moving your mouse allows you to zoom in and out.


## Known Issues

- NOT platform-independant: Code assumes that 'long' refers to 32bit integers.