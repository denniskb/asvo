/** \file */

#ifndef texture_operations_h

#define texture_operations_h

#include "texture.h"

// Defintion of constants. Right now, only textures with the following
// dimensions are supported!

#ifdef __CUDACC__
__constant__ unsigned short int TEX_WIDTH = 1024;
__constant__ unsigned short int TEX_HEIGHT = 1024;
__constant__ unsigned long int TEX_DIM = 1024 * 1024;
#else
const unsigned short int TEX_WIDTH = 1024;
const unsigned short int TEX_HEIGHT = 1024;
const unsigned long int TEX_DIM = TEX_WIDTH * TEX_HEIGHT;
#endif

/**
 * Loads a texture form a file. The texture has
 * to have a resolution of 1024x1024 pixels and be stored in
 * the raw rgb-model (3 bytes per pixel, one continous memory block).
 *
 * @param fileName The file name of the texture to be loaded.
 *
 * @return A Texture representing the image file.
 */
Texture texInit(char const * fileName);

/**
 * Does cleanup work like releasing allocated memory.
 *
 * @param tex The texture to be cleaned up.
 */
void texCleanup(Texture *tex);

#endif