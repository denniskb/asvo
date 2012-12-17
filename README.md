# Animated Sparse Voxel Octrees @ CUDA


## Introduction

Animated Sparse Voxel Octrees (ASVO) is an animation technique for sparse voxel octrees. It allows you to apply rigid transformations or skinning to voxel based 3D models.

You can find more detailed information on the [project page](http://bautembach.de/wordpress/?page_id=7).

This is the high performance CUDA implementation (aka ASVO@CUDA) of the C# prototype (aka ASVO).


## Dependencies

- Windows 7 64bit
- [CUDA Toolkit 4.2](https://developer.nvidia.com/cuda-toolkit-42-archive)
- [CUDA SDK 4.2](https://developer.nvidia.com/cuda-toolkit-42-archive)
- CUDA capable GPU with Compute Capability 2.0 or higher, 1GB VRAM or more.
- Visual Studio 2010


## Usage

After launching the program you can rotate the camera by holding down the left mouse button and moving your mouse. Holding down the right mouse button and moving your mouse allows you to zoom in and out.


## Setup

1. Change the hard-coded screen resolution in 'main.cpp' to your system's screen resolution.
2. Build and run.


## Known Issues

- Unnecessary Windows dependency (just for QueryHighPerformanceTimer and freeglut).
- Broken 32bit build.
- Outdated makefile 'standard.make'
- **NOT(!) platform-independant: Code assumes that 'long' refers to 32bit integers**.


## Roadmap

1. Remove Windows dependencies and make code platfrom-independant (remove assumptions about size and layout/endianness of types, provide make-file, etc.).
2. Port project from C to C++. Refactor code to make use of the newest CUDA API features (thrust, streams, etc.).
3. Update project to CUDA 5.0.
4. Improve/Extend algorithm.


## License

- Author: Dennis Bautembach
- ASVO@CUDA: [Creative Commons Attribution (CC BY 3.0)](http://creativecommons.org/licenses/by/3.0/deed.en_US): If you use any of the source code in your own code, please mention the original author's name in a comment or other appropriate place. ASVO@CUDA itself uses code released under CC BY 3.0 and mentions the original authors in comments. Please maintain this information.