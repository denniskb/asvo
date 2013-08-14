#pragma once

#include <thrust/device_vector.h>
#include <vector_types.h>

#include "Camera.h"
#include "Light.h"
#include "Object3D.h"
#include "VoxelData.h"

class Renderer
{
public:

	Renderer( int frameWidthInPixels, int frameHeightInPixels, bool shadowMapping );
	~Renderer();

	// TODO: Try and make obj a const param.
	void render
	(
		Object3D & obj,
		Camera const & cam,
		Light const & light,
		
		uchar4 * outColorBuffer
	);

private:

	/* Prevent Renderer from being copied */
	Renderer( Renderer const & copy );
	Renderer & operator=( Renderer const & rhs );

	static int const nTHREADS_TRAV_KERNEL = 128;
	static int const nTHREADS_DRAW_KERNEL = 128;
	static int const nTHREADS_DRAW_SHADOW_KERNEL = 192;

	int m_frameWidth;
	int m_frameHeight;
	bool m_shadowMapping;

	thrust::device_vector< BFSJob > m_dJobQueue;

	thrust::device_vector< unsigned int > m_dDepthBuffer;
	thrust::device_vector< VoxelData > m_dVoxelBuffer;
	thrust::device_vector< float > m_dShadowMap;

	cudaTextureObject_t m_tDepthBuffer;

	void rasterize
	(
		Object3D const & obj,
		Camera const & cam,
		Light const & light,
		int animationFrameIndex,

		bool shadowPass,
		
		uchar4 * outColorBuffer
	);

	void clearColorBuffer( uchar4 * dpOutColorBuffer );
	void clearDepthBuffer();
	void clearShadowMap();
	void fillJobQueue( BFSJob const * dpJobs, int jobCount );

	int resolution() const;

	/* Computes ceil( (double) nElements / nThreadsPerBlock ) */
	static int nBlocks( int nElements, int nThreadsPerBlock );
};