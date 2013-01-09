#pragma once

#include <vector_types.h>

#include "camera.h"
#include "matrix.h"
#include "object3d.h"
#include "voxeldata.h"

class Rasterizer
{
public:

	void init();
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
		uchar4 * colorBuffer,
		Object3d obj,
		Camera cam,
		Matrix lightWorldViewProjection
	);

private:

	static int const nTHREADS_CLEAR_KERNEL = 192;
	static int const nTHREADS_TRAV_KERNEL = 128;
	static int const nTHREADS_DRAW_KERNEL = 128;
	static int const nTHREADS_DRAW_SHADOW_KERNEL = 192;

	unsigned int * m_pDepthBuffer;
	VoxelData * m_pVoxelBuffer;
	float * m_pShadowMap;

	/**
	 * Encapsulates the whole render process including clearBuffers, traverse and draw.
	 * Manages the job queue and adjusts execution configurations of kernels to maximize performance.
	 *
	 * @param colorBuffer              The color buffer.
	 * @param obj                      The model to be rendered.
	 * @param cam                      The virtual camera.
	 * @param shadowPass               Determines whether the output of this pass is an image or a shadow map.
	 * @param lightWorldViewProjection light transform * model world transform * camera view transform * camera projection transform
	 */
	void render
	(
		uchar4 * colorBuffer,
		Object3d obj,
		Camera cam,
		bool shadowPass,
		Matrix lightWorldViewProjection
	);

	static int nBlocks( int nElements, int nThreadsPerBlock );
};