#pragma once

#include <vector_types.h>

#include "camera.h"
#include "matrix.h"
#include "object3d.h"
#include "voxeldata.h"

class Rasterizer
{
public:

	void init( int frameWidthInPixels, int frameHeightInPixels );
	~Rasterizer();

	/**
	 * Invokes the rasterizer on the GPU. This function exists to separate
	 * device code from host code. It is being called by the kernel-caller (host code) and
	 * itself calls the actual kernels used for rendering an asvo.
	 *
	 * @param colorBuffer              The color-buffer to output the final image into.
	 * @param obj                      The Object3d to be rendered.
	 * @param cam                      The virtual camera to be used for rendering.
	 * @param lightWorldViewProjection light matrix * obj world transform matrix * camera view matrix * camera projection matrix.
	 */
	void rasterize
	(
		Object3d & obj,
		Camera const & cam,
		int frameWidthInPixels, int frameHeightInPixels,

		bool shadowMapping,
		
		uchar4 * outColorBuffer
	);

private:

	static int const nTHREADS_CLEAR_KERNEL = 192;
	static int const nTHREADS_TRAV_KERNEL = 128;
	static int const nTHREADS_DRAW_KERNEL = 128;
	static int const nTHREADS_DRAW_SHADOW_KERNEL = 192;

	unsigned int * m_pDepthBuffer;
	VoxelData * m_pVoxelBuffer;
	float * m_pShadowMap;

	void render
	(
		Object3d const & obj,
		Camera const & cam,
		int frameWidthInPixels, int frameHeightInPixels,

		bool shadowPass,
		
		uchar4 * outColorBuffer
	);

	/* Computes ceil( (double) nElements / nThreadsPerBlock ) */
	static int nBlocks( int nElements, int nThreadsPerBlock );
};