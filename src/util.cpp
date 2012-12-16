#include "../inc/util.h"

#include <cstdio>
#include <cstdlib>

void loadImage(unsigned char **ptr, char *fileName)
{	
	*ptr = (unsigned char*) malloc(1024 * 1024 * 3);

    FILE *file;
	file = fopen(fileName, "rb");

    fread(*ptr, 1024 * 1024 * 3, 1, file);

    fclose(file);
}