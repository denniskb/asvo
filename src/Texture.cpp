#include "../inc/Texture.h"

#include <cassert>
#include <vector>

#include <vector_types.h>

Texture::Texture( char const * fileName, int width, int height ) :
	m_width( width ),
	m_height( height )
{
	// TODO: Handle errors more gracefully
	if( fileName != nullptr )
	{
		int resolution = width * height;

		FILE * file;
		file = fopen( fileName, "rb" );

		assert( file != nullptr );

		std::vector< char > hImgRGB( 3 * resolution );

		fread( & hImgRGB[ 0 ], 1, 3 * resolution, file );
		fclose(file);

		std::vector< uchar4 > hImgRGBA( resolution );
		// TODO: Improve performance by loading a 4byte word at once
		for( int i = 0; i < resolution; ++i )
		{
			hImgRGBA[ i ] = make_uchar4
			(
				hImgRGB[ 3 * i ], 
				hImgRGB[ 3 * i + 1 ], 
				hImgRGB[ 3 * i + 2 ], 
				255
			);
		}

		cudaChannelFormatDesc desc = cudaCreateChannelDesc< uchar4 >();
		cudaMallocArray( & m_pData, & desc, width, height );
		cudaMemcpyToArray( m_pData, 0, 0, & hImgRGBA[ 0 ], resolution * sizeof( uchar4 ), cudaMemcpyHostToDevice );
	}
}

Texture::Texture( Texture const & copy ) :
	m_pData( nullptr )
{
	copyFrom( copy );
}

Texture & Texture::operator=( Texture const & rhs )
{
	copyFrom( rhs );

	return * this;
}

Texture::~Texture()
{
	cudaFreeArray( m_pData );
}



int Texture::width() const
{
	return m_width;
}

int Texture::height() const
{
	return m_height;
}



cudaArray const * Texture::data() const
{
	return m_pData;
}



void Texture::copyFrom( Texture const & other )
{
	if( this == & other )
	{
		return;
	}

	m_width = other.width();
	m_height = other.height();
	int resolution = m_width * m_height;

	if( m_pData != nullptr )
	{
		cudaFreeArray( m_pData );
	}

	cudaChannelFormatDesc desc = cudaCreateChannelDesc< uchar4 >();
	cudaMallocArray( & m_pData, & desc, m_width, m_height );

	cudaMemcpyArrayToArray
	(
		m_pData, 0, 0,
		other.data(), 0, 0,
		resolution * sizeof( uchar4 ),
		cudaMemcpyDeviceToDevice
	);
}