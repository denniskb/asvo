#pragma once

#include <vector_types.h>

#include "camera.h"
#include "matrix.h"
#include "object3d.h"
#include "voxeldata.h"

class Rasterizer
{
public:

	Rasterizer( int frameWidthInPixels, int frameHeightInPixels, bool shadowMapping );
	~Rasterizer();

	void rasterize
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

	unsigned int * m_pDepthBuffer;
	VoxelData * m_pVoxelBuffer;
	float * m_pShadowMap;

	void render
	(
		Object3d const & obj,
		Camera const & cam,

		bool shadowPass,
		
		uchar4 * outColorBuffer
	);

	/* Computes ceil( (double) nElements / nThreadsPerBlock ) */
	static int nBlocks( int nElements, int nThreadsPerBlock );
};