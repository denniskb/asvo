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

	unsigned int * m_pDepthBuffer;
	VoxelData * m_pVoxelBuffer;
	float * m_pShadowMap;
};