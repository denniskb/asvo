# Animated Sparse Voxel Octrees @ CUDA


## Introduction

Animated Sparse Voxel Octrees (ASVO) is an animation technique for sparse voxel octrees. It allows you to apply rigid transformations or skinning to voxel based 3D models.

You can find more detailed information on the [project page](http://bautembach.de/wordpress/?page_id=7).

The project is divided into two sub projects: The root directory contains the high performance CUDA/C++ implementation and can be build with CMake. The 'prototype' subfolder contains the XNA/C# prototype that was used for data structure and algorithm design.


## Dependencies

### CUDA/C++ high performance implementation 
- CMAKE 3.0.0
- CUDA 7.5
- CUDA capable GPU with Compute Capability 3.0 or higher, 1GB VRAM or more.
- freeglut 3.0.0 or later
- GLEW 1.12.0 or later

### XNA/C# prototype
- [Microsoft XNA Game Studio 4.0](http://www.microsoft.com/en-gb/download/details.aspx?id=23714)
- [XNAnimation Library](http://xnanimation.codeplex.com/): This project contains a ported version of XNAnimation 0.7.0.0 BETA 3 to support XNA 4.0. Please use this copy as it contains custom changes and ASVO won't work with the official version.


## Usage

After launching the program you can rotate the camera by holding down the *left* mouse button and moving your mouse. Holding down the *right* mouse button and moving your mouse allows you to zoom in and out.


## Known Issues

- NOT platform-independant: Code assumes that 'long' refers to 32bit integers.


## License

- Author: Dennis Bautembach
- ASVO: Creative Commons Attribution (CC BY 3.0): If you use any of the source code in your own code, please mention the original author's name in a comment or other appropriate place. ASVO itself uses code released under CC BY 3.0 and mentions the original authors in comments. Please maintain this information.
- XNAnimation Library: Microsoft Public License (Ms-PL)