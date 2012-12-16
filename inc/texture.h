#ifndef texture_h

#define texture_h

#include <cuda_runtime.h>

/**
 * Represents a texture.
 */
typedef struct
{
	cudaArray *data;
} Texture;

#endif