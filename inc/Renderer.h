#pragma once

#include <thrust/device_vector.h>
#include <vector_types.h>

#include "camera.h"
#include "matrix.h"
#include "object3d.h"
#include "voxeldata.h"

class Renderer
{
public:

	Renderer( int frameWidthInPixels, int frameHeightInPixels, bool shadowMapping );

	void render
	(
		Object3d & obj,
		Camera const & cam,
		
		uchar4 * outColorBuffer
	);

private:

	static int const nTHREADS_CLEAR_KERNEL = 192;
	static int const nTHREADS_TRAV_KERNEL = 128;
	static int const nTHREADS_DRAW_KERNEL = 128;
	static int const nTHREADS_DRAW_SHADOW_KERNEL = 192;

	int m_frameWidthInPixels;
	int m_frameHeightInPixels;
	bool m_shadowMapping;

	thrust::device_vector< unsigned int > m_depthBuffer;
	thrust::device_vector< VoxelData > m_voxelBuffer;
	thrust::device_vector< float > m_shadowMap;

	void rasterize
	(
		Object3d const & obj,
		Camera const & cam,

		bool shadowPass,
		
		uchar4 * outColorBuffer
	);

	void clearColorBuffer( uchar4 * dpOutColorBuffer );
	void clearDepthBuffer();
	void clearShadowMap();

	/* Computes ceil( (double) nElements / nThreadsPerBlock ) */
	static int nBlocks( int nElements, int nThreadsPerBlock );
};