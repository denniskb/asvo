#include "../inc/texture_operations.h"
#include "../inc/texture.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

Texture texInit(char *fileName)
{
	Texture result;	

	if (fileName != NULL)
	{
		FILE *file;
		file = fopen(fileName, "rb");
		if (file != NULL)
		{
			unsigned char *temp = (unsigned char*) malloc(TEX_DIM * 3 * sizeof(unsigned char));

			fread(temp, TEX_DIM * 3, sizeof(unsigned char), file);
			fclose(file);

			uchar4 *data = (uchar4*) malloc(TEX_DIM * sizeof(uchar4));
			for (int i = 0; i < TEX_DIM; ++i)
				data[i] = make_uchar4(temp[3 * i], temp[3 * i + 1], temp[3 * i + 2], 0);

			const cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
			cudaMallocArray(&result.data, &desc, TEX_WIDTH, TEX_HEIGHT);
			cudaMemcpyToArray(result.data, 0, 0, data, TEX_DIM * sizeof(uchar4), cudaMemcpyHostToDevice);

			free(data);
			free(temp);
		}
	}

	return result;
}

void texCleanup(Texture *tex)
{
	if (tex->data != NULL)
	{
		cudaFreeArray(tex->data);
		tex->data = NULL;
	}
}