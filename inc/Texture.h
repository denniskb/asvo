#pragma once

#include <cuda_runtime.h>

class Texture
{
public:

	/* 
	 * Loads a raw rgb image (3 bytes per pixel, encoded as r, g, b).
	 */
	Texture( char const * fileName, int widthInTexels, int heightInTexels );
	Texture( Texture const & copy );
	Texture & operator=( Texture const & rhs );
	~Texture();

	int width() const;	// in texels
	int height() const;	// in texels

	cudaTextureObject_t const & textureObject() const;

private:

	cudaArray * m_pData;
	cudaTextureObject_t m_texture;
	int m_width;
	int m_height;

	void copyFrom( Texture const & other );
	void initTexture();
};